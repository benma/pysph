"""
Qt OpenGL widget, dispatching the rendering and mouse events to SPHDemo.
"""

from PySide import QtCore
from PySide.QtOpenGL import QGLWidget

from sph import FluidSimulator

class FluidWidget(QGLWidget):
    gl_initialized = QtCore.Signal()
        
    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        
    def init(self, sph_demo):
        self.sph_demo = sph_demo

    def mousePressEvent(self, event):
        buttons = event.buttons()
        if buttons & (QtCore.Qt.LeftButton | QtCore.Qt.RightButton):
            self.mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        # this method is only called when a mouse button is pressed
        pos = event.pos()
        offset = pos - self.mouse_pos
        self.mouse_pos = pos

        dx, dy = offset.x(), offset.y()
        
        buttons = event.buttons()
        if buttons & QtCore.Qt.LeftButton:
            self.sph_demo.on_mouse_move(dx, dy, 0)
        if buttons & QtCore.Qt.MiddleButton:
            self.sph_demo.on_mouse_move(dx, dy, 1)
        if buttons & QtCore.Qt.RightButton:
            self.sph_demo.on_mouse_move(dx, dy, 2)

    def wheelEvent(self, event):
        self.sph_demo.on_mouse_wheel(event.delta()/120)
    
    def initializeGL(self):
        self.sph_demo.glinit()

        # redraw at a fixed framerate
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateGL)
        fps = self.sph_demo.framerate
        self.timer.start(int(1000/float(fps)))

        self.gl_initialized.emit()
        
    def paintGL(self):
        self.sph_demo._render()        

    def resizeGL(self, w, h):
        from OpenGL.GL import glViewport
        glViewport(0, 0, w, h)
