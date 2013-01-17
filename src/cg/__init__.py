"""
Quick and easy bindings to the Cg libraries.
"""
from .cg_shader import CGVertexShader, CGFragmentShader, CGVertexFragmentShader, CGDefaultShader
from .platform import cg_platform, cg_gl_platform
from . import cg, cg_gl
