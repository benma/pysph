import sys

from PySide import QtGui
from main_window import MainWindow

if __name__ == "__main__":

    def excepthook(type, value, traceback):
        import traceback
        traceback.print_last()
        sys.exit(1)
    # PySide/PyQT just print exceptions to console and continue
    # override excepthook to exit application on error.
    sys.excepthook = excepthook
    
    app = QtGui.QApplication(sys.argv)

    # number of particles
    try:
        N = int(sys.argv[1]) #20**3
    except (IndexError, ValueError):
        print "Invalid number of particles. Use 'python main.py 8000', for example"
        exit(1)
    window = MainWindow(N)
    window.show()

    sys.exit(app.exec_())
