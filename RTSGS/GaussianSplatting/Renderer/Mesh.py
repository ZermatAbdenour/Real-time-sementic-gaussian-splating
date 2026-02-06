import ctypes
import numpy as np
from OpenGL.GL import *

class Mesh:
    def __init__(self, vertices: np.ndarray, indices: np.ndarray):
        self.vertices = vertices.astype(np.float32, copy=False)
        self.indices = indices.astype(np.uint32, copy=False)

        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.vao = glGenVertexArrays(1)

        self._setup_mesh()

    def _setup_mesh(self):
        glBindVertexArray(self.vao)

        # VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        stride = 6 * 4  # 6 

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # color
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindVertexArray(0)


    def render(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, int(self.indices.size), GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)


