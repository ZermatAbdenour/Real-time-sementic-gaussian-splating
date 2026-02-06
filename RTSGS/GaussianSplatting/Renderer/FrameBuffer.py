from __future__ import annotations
from dataclasses import dataclass
from OpenGL.GL import *

@dataclass
class FrameBuffer:
    width: int
    height: int
    with_depth_stencil: bool = True

    fbo: int = 0
    color_tex: int = 0
    depth_rbo: int = 0  # depth/stencil renderbuffer id (0 if disabled)

    def __post_init__(self) -> None:
        self.width = max(1, int(self.width))
        self.height = max(1, int(self.height))
        self._create()

    def _create(self) -> None:
        # FBO
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Color texture attachment
        self.color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_tex, 0)

        # Optional depth/stencil renderbuffer
        if self.with_depth_stencil:
            self.depth_rbo = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.width, self.height)
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_rbo
            )

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)

        # cleanup binds
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {hex(status)}")

    def bind(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

    @staticmethod
    def unbind() -> None:
        """Bind default framebuffer and restore viewport to window size."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        #glViewport(0, 0, int(window_width), int(window_height))

    def resize(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height

        # Resize color texture storage
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Resize depth storage
        if self.with_depth_stencil and self.depth_rbo:
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

    def clear(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """Clear the FBO to a color (call after bind())."""
        glDisable(GL_DEPTH_TEST)
        glClearColor(r, g, b, a)
        glClear(GL_COLOR_BUFFER_BIT | (GL_DEPTH_BUFFER_BIT if self.with_depth_stencil else 0))

    def delete(self) -> None:
        """Call on shutdown."""
        if self.depth_rbo:
            glDeleteRenderbuffers(1, [self.depth_rbo])
            self.depth_rbo = 0
        if self.color_tex:
            glDeleteTextures(1, [self.color_tex])
            self.color_tex = 0
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
            self.fbo = 0