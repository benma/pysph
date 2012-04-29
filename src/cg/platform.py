import OpenGL.platform # cgGl depends on it

from ctypes import *
cg_platform = CDLL('libCg.so')
cg_gl_platform = CDLL('libCgGL.so')
