from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GL.ARB.texture_rg import *

class RenderTarget(object):
    internal_format = GL_RGBA
    filtering = GL_LINEAR
    pixel_format = GL_RGBA
    pixel_type = GL_UNSIGNED_BYTE

    @classmethod
    def create_texture(cls, size):
        width, height = size
        
        glEnable(GL_TEXTURE_2D);
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, cls.internal_format, width, height, 0, cls.pixel_format, cls.pixel_type, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, cls.filtering)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture
    
    def __init__(self, size, clear_color=(0,0,0,1)):
        """
        size: (width, height).
        """
        self.size = size
        self.clear_color = clear_color
        width, height = size

        
        
        # create texture for color output
        self.texture = self.create_texture(size)

        # create renderbuffer for depth
        rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        # create frame buffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        # attach color texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)
        # attach depth buffer
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb)

        checkFramebufferStatus()
        #assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        """
        Bind frame buffer. Subsequent rendering goes into the render target.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_VIEWPORT_BIT | GL_ENABLE_BIT)
        glViewport(0, 0, *self.size)
        if self.clear_color:
            glClearColor(*self.clear_color)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
    def unbind(self):
        """
        Finished rendering to the render target.
        """
        glPopAttrib()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    __enter__ = bind
    def __exit__(self, *args):
        self.unbind()

class RenderTargetR32F(RenderTarget):
    """
    Texture of 32bit float elements.
    """
    internal_format = GL_R32F
    filtering = GL_NEAREST
    pixel_format = GL_RED
    pixel_type = GL_FLOAT

class RenderTargetRG32F(RenderTarget):
    internal_format = GL_RG32F
    filtering = GL_NEAREST
    pixel_format = GL_RG
    pixel_type = GL_FLOAT
