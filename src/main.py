import sys

from PySide import QtGui
from main_window import MainWindow

if __name__ == "__main__":
    if '--help' in sys.argv:
        print "Run with: python main.py [--disable-advanced-rendering] [--cg-arb] [--cg-glsl] N"
        print "\t--disable-advanced-rendering: disables the use of Cg shaders"
        print "\tChange the vertex/fragment profiles that Cg will use:"
        print "\t\t--cg-arb:\tuse arb profiles."
        print "\t\t--cg-glsl:\tuse glsl profiles."
        print "\tN: number of particles. Start with 8000 and slowly increase"
        sys.exit(0)

    def excepthook(type, value, traceback):
        import traceback
        traceback.print_last()
        sys.exit(1)
    # PySide/PyQT just print exceptions to console and continue
    # override excepthook to exit application on error.
    sys.excepthook = excepthook
    
    app = QtGui.QApplication(sys.argv)

    # number of particles (last argument in command line)
    try:
        N = int(sys.argv[-1]) #20**3
    except (IndexError, ValueError):
        print "Invalid number of particles. Use 'python main.py 8000', for example"
        print "See also: python main.py --help"
        exit(1)
    window = MainWindow(N)
    window.show()

    sys.exit(app.exec_())
