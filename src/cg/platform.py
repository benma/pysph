import os
import OpenGL.platform # cgGl depends on it

from ctypes import *

if os.name == 'nt':
    # we are on Windows
    cg_platform = CDLL('Cg.dll')
    cg_gl_platform = CDLL('CgGL.dll')
else:
    cg_platform = CDLL('libCg.so')
    cg_gl_platform = CDLL('libCgGL.so')
