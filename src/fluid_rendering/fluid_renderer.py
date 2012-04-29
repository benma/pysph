from OpenGL.GL import *
import pyopencl as cl

import numpy as np
import os

from cg import CGDefaultShader, cg_gl_platform, cg_gl
import simple

from blur_shader import BlurShader


class FluidShader(CGDefaultShader):
    
    def __init__(self, window_size, boxsize, radius, entry_vertex="vertex", entry_fragment="fragment"):
        
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open('%s/shader.cg' % cur_dir) as f:
            from mako.template import Template
            
            source = str(Template(f.read()).render(
                boxsize=boxsize,
                radius=radius,
                window_size=window_size,
                ))
        super(FluidShader, self).__init__(source, entry_vertex, entry_fragment)

        self.vertex_model_view_proj = self.get_vertex_parameter("modelViewProj")
        self.vertex_model_view = self.get_vertex_parameter("modelView")
        self.fragment_model_view = self.get_fragment_parameter("modelView")
        self.vertex_proj = self.get_vertex_parameter("proj")
        self.fragment_proj = self.get_fragment_parameter("proj")

    def set_default_parameters(self):
        cg_gl_platform.cgGLSetStateMatrixParameter(self.vertex_model_view_proj, cg_gl.CG_GL_MODELVIEW_PROJECTION_MATRIX, cg_gl.CG_GL_MATRIX_IDENTITY)
        cg_gl_platform.cgGLSetStateMatrixParameter(self.vertex_model_view, cg_gl.CG_GL_MODELVIEW_MATRIX, cg_gl.CG_GL_MATRIX_IDENTITY)
        cg_gl_platform.cgGLSetStateMatrixParameter(self.vertex_proj, cg_gl.CG_GL_PROJECTION_MATRIX, cg_gl.CG_GL_MATRIX_IDENTITY)
        cg_gl_platform.cgGLSetStateMatrixParameter(self.fragment_model_view, cg_gl.CG_GL_MODELVIEW_MATRIX, cg_gl.CG_GL_MATRIX_IDENTITY)
        cg_gl_platform.cgGLSetStateMatrixParameter(self.fragment_proj, cg_gl.CG_GL_PROJECTION_MATRIX, cg_gl.CG_GL_MATRIX_IDENTITY)

class FluidRenderer(object):
    number_of_render_modes = 3
    RENDERMODE_POINTS, RENDERMODE_BALLS, RENDERMODE_ADVANCED = range(number_of_render_modes)
    render_mode = RENDERMODE_ADVANCED
#    render_mode = RENDERMODE_BALLS

    smooth_depth = True
    blur_thickness_map = True
    smoothing_iterations = 50
    smoothing_z_contrib = 10

    # timestep for curvature flow smoothing
    # can be changed on the fly at each framerate if desired.
    smoothing_dt = 0.005

    render_mean_curvature = False
    
    def __init__(self, window_size, fluid_simulator, projection_matrix):
        """
        projection_matrix: a 4x4 numpy array describing the projection matrix in use.
        """
        self.window_size = window_size
        self.fluid_simulator = fluid_simulator
        self.projection_matrix = projection_matrix

        self.position_vbo = fluid_simulator.position_vbo
        self.N = fluid_simulator.N


        radius = fluid_simulator.h * 0.25

        # blur parameters tuned to this example. should be generalized.
        # should make the radius/sigma dependent on the largest screen-space particle size
        # so that it looks good from any distance.
        self.thickness_blur_shader = BlurShader(size=window_size, radius=20, sigma=5)
        
        self.ball_shader = FluidShader(window_size, fluid_simulator.boxsize, radius, entry_fragment="ballFragment")
        self.depth_shader = FluidShader(window_size, fluid_simulator.boxsize, radius*1.5, entry_fragment="depthFragment")
        self.final_shader = FluidShader(window_size, fluid_simulator.boxsize, radius, entry_vertex="vertexPass", entry_fragment="finalFragment")
        self.tex_shader = FluidShader(window_size, fluid_simulator.boxsize, radius, entry_vertex="vertexPass", entry_fragment="texFragment")
        self.thickness_shader = FluidShader(window_size, fluid_simulator.boxsize, radius*3, entry_fragment="thicknessFragment")

        from render_target import RenderTarget, RenderTargetR32F
        
         # alternating between the two in the smoothing iteration
        self.depth_target = RenderTargetR32F(window_size)
        self.depth2_target = RenderTargetR32F(window_size)
        
        self.test_target = RenderTarget(window_size)

        # alternating between the two for the horizontal/vertical blur pass
        self.thickness_target = RenderTargetR32F(window_size)
        self.thickness2_target = RenderTargetR32F(window_size)

        self.cl_init()

    def cl_pick_device(self):
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                if device.type == cl.device_type.GPU:
                    return device
        raise Exception("No suitable device found")

    def cl_init_context(self):
        """
        Define context and queue.
        """
        device = self.cl_pick_device()
        platform = device.platform
        additional_properties = []

        from pyopencl.tools import get_gl_sharing_context_properties
        additional_properties = get_gl_sharing_context_properties()
        
        self.ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + additional_properties)
        self.queue = cl.CommandQueue(self.ctx, device=device, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

    def cl_init(self):
        self.cl_init_context()

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open('%s/curvature_flow.cl' % cur_dir) as f:
            from mako.template import Template
            code = str(Template(f.read()).render(
                window_size=self.window_size,
                # smoothing timestep
                dt=self.smoothing_dt,
                projection_matrix=self.projection_matrix
                ))
            self.prg = cl.Program(self.ctx, code).build()
            
        self.depth_cl = cl.GLTexture(self.ctx, cl.mem_flags.READ_WRITE, GL_TEXTURE_2D, 0, self.depth_target.texture, 2)
        self.depth2_cl = cl.GLTexture(self.ctx, cl.mem_flags.READ_WRITE, GL_TEXTURE_2D, 0, self.depth2_target.texture, 2)
        self.test_cl = cl.GLTexture(self.ctx, cl.mem_flags.READ_WRITE, GL_TEXTURE_2D, 0, self.test_target.texture, 2)
        self.cl_gl_objects = [self.depth_cl, self.depth2_cl, self.test_cl]

        # local work size: (lw, lh) where lw (lh) must divide window width (height).
        lw, lh = 16, 16
        local_size_limit = min(device.max_work_group_size for device in self.ctx.devices)
        if lw*lh > local_size_limit:
            import sys
            sys.stderr.write('Warning: work group size too large, try reducing it. Until then, we are letting OpenCL implementation can pick something.\n')
            self.cl_local_size = None
        else:
            assert self.window_size[0] % lw == 0 and self.window_size[1] % lh == 0, "Window width (height) must be divisible by %i (%i)" % (lw, lh)
            self.cl_local_size = (lw, lh)

    def render(self):
        render_mode = self.render_mode
        if render_mode == self.RENDERMODE_POINTS:
            self.render_points()
        elif render_mode == self.RENDERMODE_BALLS:
            self.render_balls()
        elif render_mode == self.RENDERMODE_ADVANCED:
            self.render_advanced()

    def cycle_render_mode(self):
        self.render_mode = (self.render_mode + 1) % self.number_of_render_modes

    def _render_particles(self):
        simple.render_particles(self.position_vbo, self.N)

    def render_points(self):
        glEnable(GL_POINT_SMOOTH)
        glPointSize(3)
        glColor3f(0,0,0.8)
        simple.render_particles(self.position_vbo, self.N)

    def render_point_sprites(self, shader, enable_depth_test=True):
        """
        Render particles as screen space circles (looks like spheres, but projection is always a circle instead of an ellipse).
        """
        with shader:
            if enable_depth_test:
                glDepthMask(GL_TRUE)
                glEnable(GL_DEPTH_TEST)
            else:
                glDisable(GL_DEPTH_TEST)
            
	    # this lets use define the size of the point in the shader
	    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

            # this enables rendering of small squares
            glEnable(GL_POINT_SPRITE)
            # this enables coordinates on the small squares
            glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
            self._render_particles()
            glDisable(GL_POINT_SPRITE);

    def render_balls(self):
        self.render_point_sprites(self.ball_shader)

    def _create_thickness_map(self):
        """
        Renders the thickness map into the thickness target.
        """
        # render thickness map to thickness target
        with self.thickness_target:
            # render with additive blending (the more particles on top of each other, the thicker the fluid is).
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)
            # no depth test, we want to render them all to determine thickness.
            self.render_point_sprites(self.thickness_shader, enable_depth_test=False)
            glDisable(GL_BLEND)

        # gaussian blur of thickness map (makes the circle-artefacts coming from rendering
        # the particles as circles disappear).
        if self.blur_thickness_map:
            # blur thickness map, vertical pass
            with self.thickness2_target:
                with self.thickness_blur_shader.blur(BlurShader.VERTICAL):
                    self.render_texture(self.thickness_target.texture)
                    
            # blur thickness map, horizontal pass
            with self.thickness_target:
                with self.thickness_blur_shader.blur(BlurShader.HORIZONTAL):
                    self.render_texture(self.thickness2_target.texture)
        
    def render_advanced(self):
        """
        Render particles as a fluid surface (Simon Green's screen space rendering).
        """
        self._create_thickness_map()

        # render depth to depth target
        with self.depth_target:
            self.render_point_sprites(self.depth_shader)

        # smooth depth texture
        if self.smooth_depth:
            cl.enqueue_acquire_gl_objects(self.queue, self.cl_gl_objects)
            local_size = self.cl_local_size
            for i in xrange(self.smoothing_iterations):
                # alternate between writing to depth2_target and depth1_target
                # (can't read from and write to the same texture at the same time).
                args = (np.float32(self.smoothing_dt),
                        np.float32(self.smoothing_z_contrib),)
                self.prg.curvatureFlow(self.queue, self.window_size, local_size, self.depth_cl, self.depth2_cl, *args).wait()
                self.prg.curvatureFlow(self.queue, self.window_size, local_size, self.depth2_cl, self.depth_cl, *args).wait()
            cl.enqueue_release_gl_objects(self.queue, self.cl_gl_objects)


        if self.render_mean_curvature:
            cl.enqueue_acquire_gl_objects(self.queue, self.cl_gl_objects)
            self.prg.test(self.queue, self.window_size, self.cl_local_size, self.depth_cl, self.test_cl).wait()
            cl.enqueue_release_gl_objects(self.queue, self.cl_gl_objects)

            with self.tex_shader:
                self.render_texture(self.test_target.texture)
                
            return

        # # testing
        # cl.enqueue_acquire_gl_objects(self.queue, [self.depth_cl, self.test_cl])
        # self.prg.test(self.queue, self.depth_target.size, self.cl_local_size, self.depth_cl, self.test_cl).wait()
        # with self.tex_shader:
        #     self.render_texture(self.test_target.texture)
        #     #self.render_texture(self.depth_target.texture)
        # cl.enqueue_release_gl_objects(self.queue, [self.depth_cl, self.test_cl])
        # return

        # use alpha blending
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        # bind thickness texture
        glActiveTexture(GL_TEXTURE0+1)
        glBindTexture(GL_TEXTURE_2D, self.thickness_target.texture)
        with self.final_shader:
            self.render_texture(self.depth_target.texture)
        glActiveTexture(GL_TEXTURE0)

        glDisable(GL_BLEND)

    def render_texture(self, texture, stage=0):
        """
        Render a full screen texture.
        stage: which texture unit to make active.
        """
        glDisable(GL_CULL_FACE)

        glActiveTexture(GL_TEXTURE0+stage)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        glBegin(GL_QUADS)

        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1.0, -1.0, 0.0)

        glTexCoord2f(1.0, 0.0);
        glVertex3f(1.0, -1.0, 0.0)
        
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1.0, 1.0, 0.0)

        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1.0, 1.0, 0.0)    

        glEnd()
        
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
