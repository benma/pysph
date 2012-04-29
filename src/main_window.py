"""
Main window setup.
"""

from PySide import QtCore,QtGui
from ui import Ui_MainWindow

from fluid_rendering import FluidRenderer

class MainWindow(QtGui.QMainWindow):
    def __init__(self, N):
        super(MainWindow, self).__init__()

        # This is always the same
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # no resizing
        self.setFixedSize(self.width(), self.height())

        
        from sph_demo import SPHDemo
        self.sph_demo = SPHDemo(N, size=(self.ui.fluid.width(), self.ui.fluid.height()))

        # ui.fluid is the FluidWidget QGLWidget-widget.
        self.ui.fluid.init(self.sph_demo)
        self.ui.fluid.gl_initialized.connect(self.update_gui_from_params)

        self.setup_signals()
        statusBarInfo = QtGui.QLabel()
        statusBarInfo.setText("%s particles, initial density: %s, mass: %.02f, gas constant k: %s, timestep: %.05f" % (
                self.sph_demo.fluid_simulator.N,
                self.sph_demo.fluid_simulator.density0, 
                self.sph_demo.fluid_simulator.mass,
                self.sph_demo.fluid_simulator.k,
                self.sph_demo.fluid_simulator.dt
                ))

        self.statusBarFps = QtGui.QLabel()
        self.statusBar().addPermanentWidget(self.statusBarFps)
        self.statusBar().addWidget(statusBarInfo)

    def setup_signals(self):
        """
        Set up callbacks for the widgets.
        """

        ui = self.ui

        self.fps_timer = QtCore.QTimer()
        def callback():
            self.statusBarFps.setText("%.2f fps" % self.sph_demo.fps)
        self.fps_timer.timeout.connect(callback)
        self.fps_timer.start(1000)

        def callback(p):
            self.sph_demo.params.paused = p
        ui.paused.toggled.connect(callback)

        def callback(p):
            self.sph_demo.params.blur_thickness_map = p
        ui.blur_thickness_map.toggled.connect(callback)
        def callback(p):
            self.sph_demo.params.render_mean_curvature = p
        ui.render_mean_curvature.toggled.connect(callback)        

        def callback():
            self.sph_demo.params.render_mode = FluidRenderer.RENDERMODE_POINTS
        ui.rm_points.pressed.connect(callback)
        def callback():
            self.sph_demo.params.render_mode = FluidRenderer.RENDERMODE_BALLS
        ui.rm_balls.pressed.connect(callback)
        def callback():
            self.sph_demo.params.render_mode = FluidRenderer.RENDERMODE_ADVANCED
        ui.rm_advanced.pressed.connect(callback)
        ui.rm_advanced.toggled.connect(ui.advanced.setEnabled)

        def callback(n):
            self.sph_demo.params.smoothing_iterations = n
        ui.smoothing_iterations.valueChanged.connect(callback)

        def callback(n):
            self.sph_demo.params.smoothing_z_contrib = n
        ui.smoothing_z_contrib.valueChanged.connect(callback)

    def update_gui_from_params(self):
        params = self.sph_demo.params
        if not params.dirty:
            return
        
        ui = self.ui
        ui.paused.setOn(params.paused)
        ui.blur_thickness_map.setOn(params.blur_thickness_map)
        ui.render_mean_curvature.setOn(params.render_mean_curvature)

        { FluidRenderer.RENDERMODE_POINTS: ui.rm_points,
          FluidRenderer.RENDERMODE_BALLS: ui.rm_balls,
          FluidRenderer.RENDERMODE_ADVANCED: ui.rm_advanced }[params.render_mode].setChecked(True)

        ui.smoothing_iterations.setValue(params.smoothing_iterations)
        ui.smoothing_z_contrib.setValue(params.smoothing_z_contrib)

        ui.advanced.setEnabled(ui.rm_advanced.isChecked())

        params.dirty = False
           
    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts
        """
        key = event.key()
        params = self.sph_demo.params
        if key == QtCore.Qt.Key_R:
            self.sph_demo.fluid_simulator.set_positions()
        elif key == QtCore.Qt.Key_P:
            params.paused = not params.paused
        elif key == QtCore.Qt.Key_B:
            params.blur_thickness_map = not params.blur_thickness_map
        elif key == QtCore.Qt.Key_N:
            params.render_mode = (params.render_mode+1) % self.sph_demo.fluid_renderer.number_of_render_modes
        elif key == QtCore.Qt.Key_C:
            params.render_mean_curvature = not params.render_mean_curvature
        else:
            super(MainWindow, self).keyPressEvent(event)

        self.update_gui_from_params()
