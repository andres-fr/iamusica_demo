#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


from PySide2 import QtWidgets
#
import matplotlib.pyplot as plt
#
from matplotlib.backends.backend_qt5agg \
    import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure


# #############################################################################
# ## COMPONENTS
# #############################################################################
class MatplotlibFigure(FigureCanvas):
    """
    A FigureCanvas that takes care of creating the plt fig/ax, and also catches
    an error if collapsed to zero size.
    """

    def __init__(self):
        """
        """
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)

    def resizeEvent(self, evt):
        """
        """
        try:
            super().resizeEvent(evt)
        except ValueError:
            # ignore ValueError when collapsing tab
            pass


class TextReport(QtWidgets.QLabel):
    """
    """
    def __init__(self, title="TextReport", parent=None):
        """
        """
        super().__init__(parent)
        self.title = title

    def update(self, quantities):
        """
        :param quantities: A list of mappings in the form [(name1, val1), ...]
        """
        update_str = "\n".join(f"{name}: {val}" for name, val in quantities)
        self.setText(f"{self.title}:" + "\n" + update_str)


# #############################################################################
# ## MAIN WIDGET
# #############################################################################
class AnalysisSurface(QtWidgets.QWidget):
    """
    TODO:

    when we select a set of notes, we want to show some stats:
    timespan, intensity(min, max, mean), key(min, max)

    """
    def __init__(self,  parent=None):
        """
        """
        super().__init__(parent=parent)
        #
        lyt = QtWidgets.QHBoxLayout()
        self.setLayout(lyt)
        #
        self.hist = MatplotlibFigure()
        lyt.addWidget(self.hist)
        #
        self.stats = TextReport("Selection statistics")
        lyt.addWidget(self.stats)

    def update_hist(self, values, nbins=100, rng=(0, 1)):
        """
        """
        self.hist.ax.clear()
        self.hist.ax.hist(values, bins=nbins, range=rng)
        self.hist.draw()

    def update_stats(self, quantities):
        """
        """
        self.stats.update(quantities)
