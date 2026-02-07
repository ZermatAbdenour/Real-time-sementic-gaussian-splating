from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import dataclass
from typing import Dict, List, Tuple

from imgui_bundle import imgui

FuncKey = Tuple[str, int, str]  # (filename, line, funcname)


def _fmt_funckey(k: FuncKey) -> str:
    filename, line, funcname = k
    return f"{funcname} ({filename}:{line})"


def _is_internal_key(k: FuncKey) -> bool:
    filename, _line, funcname = k
    if filename == "~":
        return False
    if funcname.startswith("<built-in"):
        return False
    return True


@dataclass
class ProfNode:
    key: FuncKey
    ncalls: str
    tottime_ms: float
    cumtime_ms: float
    avg_ms_per_call: float


class ProfilerWindow:
    def __init__(self) -> None:
        self._prof: cProfile.Profile = cProfile.Profile()
        self._prof_enabled: bool = False  # track actual enabled state

        self._raw_text: str = ""

        self._capturing: bool = False
        self._capture_requested: bool = False
        self._frames_to_capture: int = 1
        self._frames_left: int = 0

        self.is_open: bool = False
        self.window_title: str = "Profiler (cProfile)"
        self.sort_by: str = "cumulative"
        self.limit: int = 120
        self.strip_dirs: bool = True

        self._search: str = ""
        self._show_raw: bool = False
        self._show_debug: bool = True  # helps diagnosing your issue

        self._nodes: Dict[FuncKey, ProfNode] = {}
        self._caller_to_callees: Dict[FuncKey, List[FuncKey]] = {}
        self._callee_to_callers: Dict[FuncKey, List[FuncKey]] = {}
        self._roots: List[FuncKey] = []

        self._max_depth: int = 10

    # -------------------------
    # Profiler enable/disable (paranoid)
    # -------------------------
    def _prof_enable(self) -> None:
        if not self._prof_enabled:
            self._prof.enable()
            self._prof_enabled = True

    def _prof_disable(self) -> None:
        if self._prof_enabled:
            self._prof.disable()
            self._prof_enabled = False

    # -------------------------
    # Public API
    # -------------------------
    def request_capture(self, frames: int | None = None) -> None:
        if frames is not None:
            self._frames_to_capture = max(1, int(frames))
        self._capture_requested = True

    def begin(self) -> None:
        """
        Must be called every frame.
        Hard rule: if we are not actively capturing, profiler must be disabled.
        """
        # HARD STOP: unless we are capturing, profiling must be off.
        if not self._capturing:
            self._prof_disable()

        # Start capture if requested
        if (not self._capturing) and self._capture_requested:
            self._capture_requested = False
            self._frames_left = max(1, int(self._frames_to_capture))
            self._prof.clear()
            self._capturing = True
            self._prof_enable()

    def end_collect(self) -> None:
        """
        Must be called every frame.
        """
        if not self._capturing:
            # HARD STOP again (in case user forgot begin())
            self._prof_disable()
            return

        self._frames_left -= 1
        if self._frames_left > 0:
            return

        # Finish capture now
        self._capturing = False
        self._prof_disable()
        self._collect_stats()

    def clear(self) -> None:
        self._raw_text = ""
        self._capture_requested = False
        self._frames_left = 0

        self._nodes.clear()
        self._caller_to_callees.clear()
        self._callee_to_callers.clear()
        self._roots.clear()

        # also stop profiling immediately
        self._capturing = False
        self._prof_disable()
        self._prof.clear()

    # -------------------------
    # Stats + graph
    # -------------------------
    def _collect_stats(self) -> None:
        s = io.StringIO()
        stats = pstats.Stats(self._prof, stream=s)

        if self.strip_dirs:
            stats = stats.strip_dirs()

        stats = stats.sort_stats(self.sort_by)
        stats.print_stats(self.limit)
        self._raw_text = s.getvalue()

        self._nodes.clear()
        self._caller_to_callees.clear()
        self._callee_to_callers.clear()
        self._roots.clear()

        for func_key, (cc, nc, tt, ct, _callers) in stats.stats.items():
            prim_calls = int(nc) if int(nc) > 0 else 0
            tottime_ms = float(tt) * 1000.0
            cumtime_ms = float(ct) * 1000.0
            avg_ms = (cumtime_ms / prim_calls) if prim_calls > 0 else 0.0
            ncalls = f"{nc}" if cc == nc else f"{nc}/{cc}"

            self._nodes[func_key] = ProfNode(
                key=func_key,
                ncalls=ncalls,
                tottime_ms=tottime_ms,
                cumtime_ms=cumtime_ms,
                avg_ms_per_call=avg_ms,
            )

        all_funcs = set(self._nodes.keys())
        internal_funcs = {k for k in all_funcs if _is_internal_key(k)}

        for callee_key, (_cc, _nc, _tt, _ct, callers) in stats.stats.items():
            for caller_key in callers.keys():
                self._callee_to_callers.setdefault(callee_key, []).append(caller_key)
                if caller_key in internal_funcs and callee_key in internal_funcs:
                    self._caller_to_callees.setdefault(caller_key, []).append(callee_key)

        def metric(k: FuncKey) -> float:
            n = self._nodes.get(k)
            if not n:
                return 0.0
            return n.tottime_ms if self.sort_by == "tottime" else n.cumtime_ms

        for caller, callees in list(self._caller_to_callees.items()):
            seen: set[FuncKey] = set()
            uniq: List[FuncKey] = []
            for c in callees:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            uniq.sort(key=metric, reverse=True)
            self._caller_to_callees[caller] = uniq

        roots: List[FuncKey] = []
        for f in internal_funcs:
            callers = self._callee_to_callers.get(f, [])
            internal_callers = [c for c in callers if c in internal_funcs]
            if not internal_callers:
                roots.append(f)

        roots.sort(key=metric, reverse=True)
        if not roots:
            keys = list(internal_funcs)
            keys.sort(key=metric, reverse=True)
            roots = keys[:25]
        self._roots = roots

    # -------------------------
    # UI
    # -------------------------
    def _render_node(self, key: FuncKey, depth: int, needle: str, visited: set[FuncKey]) -> bool:
        if depth > self._max_depth:
            return False

        node = self._nodes.get(key)
        label = _fmt_funckey(key)
        label_lower = label.lower()

        children = self._caller_to_callees.get(key, [])
        if key in visited:
            children = []

        self_match = (not needle) or (needle in label_lower)
        child_matches = False
        if needle:
            for ch in children:
                if needle in _fmt_funckey(ch).lower():
                    child_matches = True
                    break

        if not (self_match or child_matches or not needle):
            return False

        has_children = len(children) > 0
        flags = imgui.TreeNodeFlags_.span_avail_width
        if not has_children:
            flags |= imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open

        if node:
            label_ui = (
                f"{label} | calls={node.ncalls}"
                f" | tot={node.tottime_ms:.3f}ms"
                f" | cum={node.cumtime_ms:.3f}ms"
            )
        else:
            label_ui = label

        opened = imgui.tree_node_ex(label_ui, flags)
        if has_children and opened:
            visited2 = set(visited)
            visited2.add(key)
            for ch in children:
                self._render_node(ch, depth + 1, needle, visited2)
            imgui.tree_pop()

        return True

    def render(self) -> None:
        if not self.is_open:
            return

        imgui.begin(self.window_title)

        if self._show_debug:
            imgui.text_disabled(
                f"debug: prof_enabled={self._prof_enabled} capturing={self._capturing} "
                f"requested={self._capture_requested} frames_left={self._frames_left}"
            )

        if self._capturing:
            imgui.text_colored((1.0, 0.6, 0.2, 1.0), f"Capturing… ({self._frames_left} frames left)")
        elif self._capture_requested:
            imgui.text_colored((0.8, 0.8, 0.2, 1.0), f"Will capture next {self._frames_to_capture} frame(s)")
        else:
            imgui.text_disabled("Idle")

        changed, frames = imgui.input_int("Frames to capture", int(self._frames_to_capture))
        if changed:
            self._frames_to_capture = max(1, int(frames))

        if imgui.button("Capture"):
            self.request_capture()

        imgui.same_line()
        if imgui.button("Clear"):
            self.clear()

        imgui.separator()

        changed, txt = imgui.input_text("Search function", self._search, 256)
        if changed:
            self._search = txt
        imgui.same_line()
        if imgui.button("x"):
            self._search = ""

        imgui.separator()

        if not self._nodes:
            imgui.text_disabled("No capture yet. Set frames and click Capture.")
            imgui.end()
            return

        needle = self._search.strip().lower()

        imgui.text_disabled("Call tree (caller -> callee), ordered by cumulative time:")
        imgui.begin_child("call_tree", (0, 300), True)

        roots_to_render = self._roots

        # --- SEARCH FIX: if needle, make matching functions temporary roots ---
        if needle:
            matching_keys = [k for k in self._nodes if needle in _fmt_funckey(k).lower()]
            # Sort by metric so the most expensive ones appear on top
            matching_keys.sort(key=lambda k: self._nodes[k].cumtime_ms if self.sort_by == "cumulative" else self._nodes[k].tottime_ms, reverse=True)
            roots_to_render = matching_keys

        for root in roots_to_render:
            self._render_node(root, 0, needle, visited=set())
        imgui.end_child()

        changed, self._show_raw = imgui.checkbox("Show raw pstats output", self._show_raw)
        if self._show_raw:
            imgui.begin_child("raw_pstats", (0, 220), True)
            imgui.text_unformatted(self._raw_text)
            imgui.end_child()

        imgui.end()
