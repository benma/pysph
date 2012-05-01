"""
Quick and easy wrappers to create vertex/fragment shaders.
"""
import sys
from ctypes import *

from cg import *
from cg_gl import *

from platform import cg_platform as cg, cg_gl_platform as cg_gl

cg_gl.cgGLSetDebugMode(CG_FALSE)
cg.cgGetParameterName.restype = c_char_p
cg.cgGetProfileString.restype = c_char_p
cg.cgGetLastErrorString.restype = c_char_p
cg.cgGetLastListing.restype = c_char_p

def check_for_cg_error(context, msg):
    error = CGerror()
       
    string = cg.cgGetLastErrorString(byref(error))
    if error.value != CG_NO_ERROR:
        sys.stderr.write("%s: %s\n" % (msg, string))
        if error.value == CG_COMPILER_ERROR:
            sys.stderr.write("%s\n" % cg.cgGetLastListing(context))
        sys.exit()

def create_context(shader_type):
    context = cg.cgCreateContext()
    check_for_cg_error(context, "creating context")
    
    cg.cgSetParameterSettingMode(context, CG_DEFERRED_PARAMETER_SETTING)
    check_for_cg_error(context, "fu")
    profile = cg_gl.cgGLGetLatestProfile(shader_type)
    #print cg.cgGetProfileString(profile)
    cg_gl.cgGLSetOptimalOptions(profile)
    check_for_cg_error(context, "selecting profile")

    return context, profile

class CGParameter(object):
    def __init__(self, name, parameter):
        self.name = name
        self.parameter = parameter
        # for ctypes 
        self._as_parameter_ = parameter

    def setf(self, value):
        cg_gl.cgGLSetParameter1f(self, c_float(value))
    def set2f(self, x, y):
        cg_gl.cgGLSetParameter2f(self, c_float(x), c_float(y))
    def set3f(self, x, y, z):
        cg_gl.cgGLSetParameter3f(self, c_float(x), c_float(y), c_float(z))
    def set4f(self, x, y, z, w):
        cg_gl.cgGLSetParameter4f(self, c_float(x), c_float(y), c_float(z), c_float(w))

class _CGShader(object):
    """
    Base class for vertex/fragment shader. Not to be used directly.
    Use one of CGVertexShader, CGFragmentShader or CGVertexFragmentShader.
    """
    vertex = False
    fragment = False
    def __init__(self, code, entry=None):
        # for now: vertex and fragment shader in same source
        assert self.vertex ^ self.fragment, "Use one of CGVertexShader, CGFragmentShader or CGVertexFragmentShader."
        
        self.entry = entry or self.default_entry
        self.context, self.profile = create_context(CG_GL_VERTEX if self.vertex else CG_GL_FRAGMENT)

        program = cg.cgCreateProgram(
            self.context, # Cg runtime context
            CG_SOURCE, # Program in human-readable form
            code, # string of source code
            self.profile, # Profile: OpenGL ARB vertex program
            entry,  # Entry function name
            None # No extra compiler options
            )
        check_for_cg_error(self.context, "creating program from string")
        cg_gl.cgGLLoadProgram(program)
        check_for_cg_error(self.context, "loading program")
        self.program = program

        self.error_prefix = '%s shader' % ("vertex" if self.vertex else "fragment")

    def check_error(self, msg):
        check_for_cg_error(self.context, '%s: %s' % (self.error_prefix, msg))

    def bind(self):
        cg_gl.cgGLBindProgram(self.program)
        self.check_error("binding program")
        cg_gl.cgGLEnableProfile(self.profile)
        self.check_error("enabling profile")

    def unbind(self):
        cg_gl.cgGLDisableProfile(self.profile)
        self.check_error("disabling profile")

    def get_parameter(self, name):
        p = cg.cgGetNamedParameter(self.program, name)
        self.check_error("could not get %r parameter" % name)
        return CGParameter(name, p)

    def update_parameters(self):
        """
        Force update of the parameters. Note that parameter updates automatically
        happen during bind (Cg Reference Manual, see cgSetParameterSettingMode).
        """
        cg.cgUpdateProgramParameters(self.program)
        self.check_error("updating parameters")

    def __enter__(self):
        self.bind()
        
    def __exit__(self, *args):
        self.unbind()

class CGVertexShader(_CGShader):
    vertex = True
    default_entry = "vertex"

class CGFragmentShader(_CGShader):
    fragment = True
    default_entry = "fragment"

class CGVertexFragmentShader(object):
    """
    Combined vertex/fragment shader. When bound, both the vertex shader and the fragment shader are bound.
    """
    def __init__(self, code, entry_vertex="vertex", entry_fragment="fragment"):
        assert entry_vertex and entry_fragment
        
        self.vertex_shader = CGVertexShader(code, entry_vertex)
        self.fragment_shader = CGFragmentShader(code, entry_fragment)

    def bind(self):
        self.vertex_shader.bind()
        self.fragment_shader.bind()

    def unbind(self):
        self.vertex_shader.unbind()
        self.fragment_shader.unbind()

    def get_vertex_parameter(self, name):
        return self.vertex_shader.get_parameter(name)

    def get_fragment_parameter(self, name):
        return self.fragment_shader.get_parameter(name)

    def update_parameters(self):
        self.vertex_shader.update_parameters()
        self.fragment_shader.update_parameters()
        
    def __enter__(self):
        self.bind()
        
    def __exit__(self, *args):
        self.unbind()

class CGDefaultShader(CGVertexFragmentShader):
    """
    A CGShader which sets (using the set_default_parameters method) all parameters automatically when its bound.
    """
    def __init__(self, *args, **kwargs):
        super(CGDefaultShader, self).__init__(*args, **kwargs)

    def set_default_parameters(self):
        # override this in subclasses
        pass

    def bind(self):
        self.set_default_parameters()
        super(CGDefaultShader, self).bind()
