"""
Quick and easy wrappers to create vertex/fragment shaders.
"""
import sys
from ctypes import *

from cg import *
from cg_gl import *

from platform import cg_platform as cg, cg_gl_platform as cg_gl

cg_gl.cgGLSetDebugMode(CG_FALSE)
cg.cgCreateProgram.restype = CGprogram
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

def memoize(f):
    cache = {}
    def new_f(*args):
        if args in cache:
            return cache[args]
        return cache.setdefault(args, f(*args))
    return new_f

@memoize
def _create_context():
    context = cg.cgCreateContext()
    check_for_cg_error(context, "creating context")
    cg.cgSetParameterSettingMode(context, CG_DEFERRED_PARAMETER_SETTING)
    return context

@memoize
def create_profile(shader_type):
    context = _create_context()
    if '--cg-glsl' in sys.argv:
        # force glsl profiles
        profile = { CG_GL_VERTEX: CG_PROFILE_GLSLV,
                    CG_GL_FRAGMENT: CG_PROFILE_GLSLF }[shader_type]
    elif '--cg-arb' in sys.argv:
        # force arg profiles
        profile = { CG_GL_VERTEX: CG_PROFILE_ARBVP1,
                    CG_GL_FRAGMENT: CG_PROFILE_ARBFP1 }[shader_type]
    else:
        profile = cg_gl.cgGLGetLatestProfile(shader_type)
    
    print "profile: ", cg.cgGetProfileString(profile)
    cg_gl.cgGLSetOptimalOptions(profile)
    check_for_cg_error(context, "selecting profile")

    return profile

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
        self.context = _create_context()
        self.profile = create_profile(CG_GL_VERTEX if self.vertex else CG_GL_FRAGMENT)

        program = cg.cgCreateProgram(
            self.context, # Cg runtime context
            CG_SOURCE, # Program in human-readable form
            code, # string of source code
            self.profile, # Profile: OpenGL ARB vertex program
            entry,  # Entry function name
            None # No extra compiler options
            )
        
        self.error_prefix = '%s shader, entry = %s' % ("vertex" if self.vertex else "fragment", entry)

        self.check_error("creating program from string")
        cg_gl.cgGLLoadProgram(program)
        self.check_error("loading program")
        self.program = program

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
    Combined vertex/fragment shader. 
    """
    def __init__(self, code, entry_vertex="vertex", entry_fragment="fragment"):
        assert entry_vertex and entry_fragment
        vertex_shader = CGVertexShader(code, entry_vertex)
        fragment_shader = CGFragmentShader(code, entry_fragment)
        self.context = _create_context()
        self.vertex_profile = vertex_shader.profile
        self.fragment_profile = fragment_shader.profile
        self.program = cg.cgCombinePrograms2(vertex_shader.program, fragment_shader.program)
        check_for_cg_error(self.context, "combine programs")
        cg.cgDestroyProgram(vertex_shader.program)
        cg.cgDestroyProgram(fragment_shader.program)
        cg_gl.cgGLLoadProgram(self.program)
        check_for_cg_error(self.context, "load combined program")

    def check_error(self, msg):
        check_for_cg_error(self.context, msg)

    def bind(self):
        cg_gl.cgGLBindProgram(self.program)
        self.check_error("binding program")
        cg_gl.cgGLEnableProfile(self.vertex_profile)
        self.check_error("enable vertex profile")
        cg_gl.cgGLEnableProfile(self.fragment_profile)
        self.check_error("enable fragment profile")

    def unbind(self):
        cg_gl.cgGLDisableProfile(self.vertex_profile)
        self.check_error("disable vertex profile")
        cg_gl.cgGLDisableProfile(self.fragment_profile)
        self.check_error("disable fragment profile")
        cg_gl.cgGLUnbindProgram(self.vertex_profile)
        cg_gl.cgGLUnbindProgram(self.fragment_profile)

    def get_parameter(self, name):
        p = cg.cgGetNamedParameter(self.program, name)
        self.check_error("could not get %r parameter" % name)
        return CGParameter(name, p)

    def update_parameters(self):
        cg.cgUpdateProgramParameters(self.program)
        self.check_error("updating parameters")
        
    def __enter__(self):
        self.bind()
        self.update_parameters()
        
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
