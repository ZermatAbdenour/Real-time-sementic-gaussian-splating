import cv2
import numpy as np
from OpenGL import GL
from imgui_bundle import imgui


class ImageWidget:
    """
    ImGui widget that displays a cv2-loaded image using imgui_bundle.

    Note: imgui_bundle's imgui.image expects an ImTextureRef (not a raw int texture id).
    This build supports constructing it directly: imgui.ImTextureRef(tex_id).
    """

    def __init__(self, rgb: np.ndarray):
        self._tex_id: int | None = None
        self._tex_ref: imgui.ImTextureRef | None = None

        rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGBA)
        rgba = np.ascontiguousarray(rgba)
        self._img_h, self._img_w = rgba.shape[:2]
        self._upload_rgba_to_texture(rgba, self._img_w, self._img_h)

    def set_image_rgb(self, rgb: np.ndarray):
        if rgb is None or rgb.size == 0:
            return

        rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGBA)
        rgba = np.ascontiguousarray(rgba)
        h, w = rgba.shape[:2]
        self._img_w, self._img_h = w, h
        self._upload_rgba_to_texture(rgba, w, h)

    def _upload_rgba_to_texture(self, rgba: np.ndarray, w: int, h: int):
        if self._tex_id is None:
            self._tex_id = int(GL.glGenTextures(1))
            # Create the ImGui texture reference once
            self._tex_ref = imgui.ImTextureRef(int(self._tex_id))

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            w,
            h,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            rgba,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def draw(self, fit_to_window: bool = True):
        self.draw_contents(fit_to_window=fit_to_window)

    def draw_contents(self, fit_to_window: bool = True):
        if self._tex_ref is None or self._img_w <= 0 or self._img_h <= 0:
            imgui.text("No image loaded.")
            return

        if fit_to_window:
            avail_w, avail_h = imgui.get_content_region_avail()
            if avail_w <= 0 or avail_h <= 0:
                disp_w, disp_h = float(self._img_w), float(self._img_h)
            else:
                scale = min(avail_w / self._img_w, avail_h / self._img_h)
                disp_w = max(1.0, self._img_w * scale)
                disp_h = max(1.0, self._img_h * scale)
        else:
            disp_w, disp_h = float(self._img_w), float(self._img_h)

        imgui.image(self._tex_ref, (disp_w, disp_h))

    def destroy(self):
        if self._tex_id is not None:
            try:
                GL.glDeleteTextures([self._tex_id])
            except Exception:
                pass
        self._tex_id = None
        self._tex_ref = None