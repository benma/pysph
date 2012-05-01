from sph import FluidSimulator

from OpenGL.GL import *
import base_demo

class SPHDemo(base_demo.BaseDemo):
    
    def __init__(self, N=1000, size=(800,800), enable_advanced_rendering=True):
        """
        enable_advanced_rendering: if False, disable advanced rendering using Cg (balls/advanced rendering).
        """
        super(SPHDemo, self).__init__(size=size)
        
        self.N = N

        self.initrans = [-6., -4., -20.]

        # box size in x, y and z dimension.
        self.boxsize = (10, 10, 10)
        self.fluid_simulator = FluidSimulator(self.N, self.boxsize, gl_interop=True)
        self.framerate = 60

        self.enable_advanced_rendering = enable_advanced_rendering

    def glinit(self):
        super(SPHDemo, self).glinit()

        # fluid simulator needs vertex buffer objects, this is why this needs to be initialized here
        self.fluid_simulator.cl_init()
        if self.enable_advanced_rendering:
            from fluid_rendering.fluid_renderer import FluidRenderer
            self.fluid_renderer = fluid_renderer = FluidRenderer(self.size, self.fluid_simulator, self.projection_matrix)

        class Params(object):
            def __init__(self, **kwargs):
                self.dirty = False
                for param, initial_value in kwargs.iteritems():
                    setattr(self, param, initial_value)
            
            @property
            def paused(self):
                return self._paused

            @paused.setter
            def paused(self, value):
                self._paused = value
                self.dirty = True

            if self.enable_advanced_rendering:
                @property
                def blur_thickness_map(self):
                    return fluid_renderer.blur_thickness_map

                @blur_thickness_map.setter
                def blur_thickness_map(self, value):
                    fluid_renderer.blur_thickness_map = value
                    self.dirty = True

                @property
                def render_mode(self):
                    return fluid_renderer.render_mode

                @render_mode.setter
                def render_mode(self, value):
                    fluid_renderer.render_mode = value
                    self.dirty = True

                @property
                def smoothing_iterations(self):
                    return fluid_renderer.smoothing_iterations

                @smoothing_iterations.setter
                def smoothing_iterations(self, value):
                    fluid_renderer.smoothing_iterations = max(0, min(100, value))
                    self.dirty = True

                @property
                def smoothing_z_contrib(self):
                    return fluid_renderer.smoothing_z_contrib

                @smoothing_z_contrib.setter
                def smoothing_z_contrib(self, value):
                    fluid_renderer.smoothing_z_contrib = max(0, min(100, value))
                    self.dirty = True

                @property
                def render_mean_curvature(self):
                    return fluid_renderer.render_mean_curvature

                @render_mean_curvature.setter
                def render_mean_curvature(self, value):
                    fluid_renderer.render_mean_curvature = value
                    self.dirty = True
                
        initial_params = {
            'paused': False
        }
        if self.enable_advanced_rendering:
            initial_params.update(blur_thickness_map=True,
                                  render_mode=FluidRenderer.RENDERMODE_BALLS,
                                  smoothing_iterations=50,
                                  smoothing_z_contrib=10,
                                  render_mean_curvature=False)
        self.params = Params(**initial_params)

    def render_box(self):
        
        glColor3f(0, 0, 0)
        pairs = [((0., 0., 0.), (0., 1., 0.)),
                 ((0., 1., 0.), (0., 1., 1.)),
                 ((0., 1., 1.), (0., 0., 1.)),
                 ((0., 0., 1.), (0., 0., 0.)),
        
                 ((1., 0., 0.), (1., 1., 0.)),
                 ((1., 1., 0.), (1., 1., 1.)),
                 ((1., 1., 1.), (1., 0., 1.)),
                 ((1., 0., 1.), (1., 0., 0.)),

                 ((0., 0., 0.), (1., 0., 0.)),
                 ((0., 1., 0.), (1., 1., 0.)),
                 ((0., 1., 1.), (1., 1., 1.)),
                 ((0., 0., 1.), (1., 0., 1.))]
        
        for p1, p2 in pairs:
            self.draw_line(p1, p2)

    def render(self):
        
        ## render box
        glMatrixMode(GL_MODELVIEW)
        # second mouse transform
        self.mouse_transform()
        # first scale
        glScale(*self.boxsize)
        self.render_box()

        ## do the simulation
        if not self.params.paused:
            # n = 1/(30 * dt), if exectuted n times, we would simulate one second of physics in 30 frames.
            # i.e., if we screenshoted this and made a movie at framerate 30 fps, we would have a real time simulation.
            # TODO: since this number is rounded, it is not entirely accurate.
            # improve with the method explained here: http://gafferongames.com/game-physics/fix-your-timestep/

            for i in xrange(int(1/(self.framerate*self.fluid_simulator.dt))):
               self.fluid_simulator.step()
            #self.fluid_simulator.step()

        ## render particles
        self.mouse_transform()
        if self.enable_advanced_rendering:
            self.fluid_renderer.render()
        else:
            from fluid_rendering import simple
            simple.render_points(self.fluid_simulator.position_vbo, self.fluid_simulator.N)

