import torch
import numpy as np
from OpenGL.GL import *
from OpenGL import GL
from .FrameBuffer import FrameBuffer
import RTSGS.GaussianSplatting.Renderer.Resources as res
from .Camera import Camera
from imgui_bundle import imgui

class Renderer:
    def __init__(self, pcd):
        # Initialize the resources
        res.init_resources()

        self.fb = FrameBuffer(width=800, height=600)
        self.pcd = pcd
        self.vbo_capacity_bytes = 0
        # Setup OpenGL buffers
        self.vbo = None
        self.vao = None
        self._initialized = False
        
        #camera setup
        self.camera = Camera()

        #Opengl 
        # Enable depth testing for 3D points
        self.fb.bind()
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_PROGRAM_POINT_SIZE) 
        res.simple_shader.use()
        self.fb.unbind()

        self.pcd_added_size = 0

        print("GL_VENDOR  :", glGetString(GL_VENDOR).decode())
        print("GL_RENDERER:", glGetString(GL_RENDERER).decode())
        print("GL_VERSION :", glGetString(GL_VERSION).decode())
        

    def _initialize_pcd_rendering(self):

        if self._initialized or self.pcd.all_points is None or self.pcd.all_points.numel() == 0:
            return

        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # allocate exactly current size
        self.vbo_capacity_bytes = (
            self.pcd.all_points.numel() * self.pcd.all_points.element_size()
        )
        glBufferData(GL_ARRAY_BUFFER, self.vbo_capacity_bytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Setup VAO
        self.vao = glGenVertexArrays(1)
        stride = 6 * 4  # 6 floats * 4 bytes
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # position (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE,
            stride,
            ctypes.c_void_p(0)
        )

        # color (location = 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE,
            stride,
            ctypes.c_void_p(12)
        )

        glBindVertexArray(0)
        self._initialized = True
        self.update_vbo(self.pcd.all_points,self.pcd.all_colors)

    def update_vbo(self, positions,colors):
        if positions is None or positions.numel() == 0:
            return

        positions_data = positions.detach().cpu().numpy().astype(np.float32)
        colors_data = colors.detach().cpu().numpy().astype(np.float32)
        #colors_data[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        interleaved = np.hstack([positions_data, colors_data])
        required_bytes = interleaved.nbytes

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Reallocate if needed
        if required_bytes > self.vbo_capacity_bytes:
            self.vbo_capacity_bytes = required_bytes
            glBufferData(GL_ARRAY_BUFFER, self.vbo_capacity_bytes, None, GL_DYNAMIC_DRAW)

        glBufferSubData(GL_ARRAY_BUFFER, 0, required_bytes, interleaved)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def Render(self):
        self.fb.bind()

        glViewport(0, 0, self.fb.width, self.fb.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.08, 0.1, 0.13, 1.0)
        self.render_pcd()
        
        self.fb.unbind()

    def render_pcd(self):
        # Initialize if needed
        self._initialize_pcd_rendering()
        if not self._initialized :
            return
        # Set point size
        

        # Updat VBO with current point data
        if(self.pcd_added_size< self.pcd.all_points.shape[0]):
            self.update_vbo(self.pcd.all_points,self.pcd.all_colors)
        # Render point
        
        self.camera.update_view()
        #update uniforms
        glUniformMatrix4fv(
            glGetUniformLocation(res.simple_shader.program, 'u_view'),
            1,                         
            GL_FALSE,                   
            self.camera.view           
        )
        glUniformMatrix4fv(
            glGetUniformLocation(res.simple_shader.program, 'u_projection'),
            1,                         
            GL_FALSE,                   
            self.camera.projection           
        )

        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, int(self.pcd.all_points.shape[0]))
        glBindVertexArray(0)


    def cleanup(self):
        """Release resources"""
        if self._initialized:
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])

    def on_resize(self):
        self.camera.update_projection(self.fb)
        