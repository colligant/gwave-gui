
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class MplWidget(QtWidgets.QWidget):
	def __init__(self, parent = None):
		QtWidgets.QWidget.__init__(self, parent)
		self.canvas = MplCanvas()
		self.vbl = QtWidgets.QVBoxLayout()
		self.vbl.addWidget(self.canvas)
		self.setLayout(self.vbl)


class MplCanvas(Canvas):
	"""Class to represent the FigureCanvas widget"""
	def __init__(self):
		self.fig = Figure()
		self.ax = self.fig.add_subplot(111)

		Canvas.__init__(self, self.fig)
		Canvas.setSizePolicy(self,
		QtWidgets.QSizePolicy.Expanding,
		QtWidgets.QSizePolicy.Expanding)
		Canvas.updateGeometry(self)





