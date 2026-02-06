from pathlib import Path
from OpenGL.GL import *


class Shader:
    def __init__(self, vertex_shader_path: str, fragment_shader_path: str):
        vertex_src = self._read_file(vertex_shader_path)
        fragment_src = self._read_file(fragment_shader_path)
        self.program = self._create_program(vertex_src, fragment_src)

    def use(self):
        glUseProgram(self.program)

    @staticmethod
    def _compile_shader(source: str, shader_type) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not status:
            log = glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            glDeleteShader(shader)
            raise RuntimeError(f"Shader compile failed:\n{log}")
        return shader

    @classmethod
    def _create_program(cls, vertex_src: str, fragment_src: str) -> int:
        vs = cls._compile_shader(vertex_src, GL_VERTEX_SHADER)
        fs = cls._compile_shader(fragment_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)

        status = glGetProgramiv(program, GL_LINK_STATUS)
        if not status:
            log = glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            glDeleteProgram(program)
            raise RuntimeError(f"Program link failed:\n{log}")

        glDetachShader(program, vs)
        glDetachShader(program, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)
        return program

    @staticmethod
    def _read_file(path: str) -> str:
        base_dir = Path(__file__).parent
        with open((base_dir/path).resolve(), "r", encoding="utf-8") as f:
            return f.read()
