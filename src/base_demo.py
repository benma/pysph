from OpenGL.GL import *
import time
import numpy as np

class BaseDemo(object):
    def __init__(self, size=(800,800)):
        self.size = size
        self.aspect_ratio = self.size[0] / float(self.size[1])
        
        self.rotate = [0,0]
        self.translate = [0., 0., 0.]
        self.initrans = [0., 0., -5.]

        self._time = time.time()
        self._frames = 0
        self.fps = 0

        # desired framerate of simulation
        self.framerate = 60

    def glinit(self):
        """
        Call after OpenGL context (window) creation.
        """
        glClearColor(150/255., 150/255., 150/255., 255/255.)
        glClearDepth(1)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE);

        from math import pi
        self.projection_matrix = self.create_projection_matrix(pi/2.5, self.aspect_ratio, 0.1, 101)

    @property
    def projection_matrix(self):
        return self._projection_matrix
    
    @projection_matrix.setter
    def projection_matrix(self, m):
        """
        Sets the projection matrix. Use create_projection_matrix to create one.
        """
        glMatrixMode(GL_PROJECTION)
        self._projection_matrix = m
        # 'F' means column major, which opengl wants.
        glLoadMatrixf(self._projection_matrix.flatten('F'))
        
        glMatrixMode(GL_MODELVIEW)        
        

    def create_projection_matrix(self, fov_y, aspect_ratio, near, far, right_handed=True):
        from math import radians, tan

        h = 1.0 / tan(fov_y*0.5)
        w = h / aspect_ratio

        if right_handed:
            return np.array([[w,0,0,0],
                             [0,h,0,0],
                             [0,0,far/float(near-far),near*far/float(near-far)],
                             [0,0,-1,0]])
        else:
            return np.array([[w,0,0,0],
                             [0,h,0,0],
                             [0,0,far/float(far-near),near*far/float(near-far)],
                             [0,0,1,0]])
        
        return m


    def mouse_transform(self, identity=True):
        #handle mouse transformations
        glMatrixMode(GL_MODELVIEW)
        if identity:
            glLoadIdentity()

        # translate again
        glTranslatef(*self.initrans)
        # then rotate
        glRotatef(self.rotate[0], 1, 0, 0)
        glRotatef(self.rotate[1], 0, 1, 0)
        # first translate
        glTranslatef(*self.translate)


    def on_mouse_wheel(self, delta):
        self.translate[2] += delta*.5

    def on_mouse_move(self, dx, dy, button):
        """
        button: 0 = left, 1 = middle, 2 = right
        """
        if button == 0:
            self.rotate[0] += dy * .2
            self.rotate[1] += dx * .2
        elif button == 2:
            self.translate[0] += dx * .03
            self.translate[1] -= dy * .03

    def draw_axes(self):
        """ Draw unit axes """
        for axis in ((1,0,0),
                     (0,1,0),
                     (0,0,1)):
            glColor3f(*axis)
            self.draw_line((0,0,0), axis)

    def draw_line(self, v1, v2):
        glBegin(GL_LINES)
        glVertex3f(*v1)
        glVertex3f(*v2)
        glEnd()

    def _render(self):
        # calc fps
        t = time.time()
        dt = t - self._time
        if dt >= 1:
            self.fps = self._frames/float(dt)
            self._frames = 0
            self._time = t
        else:
            self._frames += 1


        glFlush()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.render()

    def render(self):
        pass

