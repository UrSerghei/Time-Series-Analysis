from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

class plot_class(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.canvas, self)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.toolbar)

        self.canvas.axis1 = self.canvas.figure.add_subplot(111)
        self.canvas.axis1 = self.canvas.figure.subplots_adjust(top=0.936, bottom=0.104, left=0.047, right=0.981, hspace=0.2, wspace=0.2)
        self.canvas.figure.set_facecolor("xkcd:pale peach")
        self.setLayout(vertical_layout)
        