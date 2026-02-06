from .Shader import Shader
from .Mesh import Mesh
import numpy as np
#shaders
simple_shader = None


#primetives
quad = None

def init_resources():
    #shaders
    global simple_shader
    simple_shader = Shader('./Shaders/point_vertex.glsl','./Shaders/point_fragment.glsl')

    #primitives
    global quad
    quad = Mesh(
        np.array([
            -0.5,-0.5,0.0,
            -0.5,0.5,0.0,
            0.5,0.5,0.0,
            0.5,-0.5,0.0
        ]).astype(np.float32),
        np.array([
            0,1,2,
            0,2,3
        ]).astype(np.float32)
    )