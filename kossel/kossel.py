import os
import signal
import sys
import h5py
import lmfit
import numpy as np
import scipy.ndimage as snd
from scipy.spatial.transform import Rotation
import skimage.morphology as skm
import kosselui
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem

import numexpr as ne

axes = [
    [-1, -1, -1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [1, 1, 1],
    [-2, -2, 0],
    [-2, 0, -2],
    [-2, 0, 2],
    [-2, 2, 0],
    [0, -2, -2],
    [0, -2, 2],
    [0, 2, -2],
    [0, 2, 2],
    [2, -2, 0],
    [2, 0, -2],
    [2, 0, 2],
    [2, 2, 0],
    [-3, -1, 1],
    [-3, 1, -1],
    [-1, -3, 1],
    [-1, -1, 3],
    [-1, 1, -3],
    [-1, 3, -1],
    [1, -3, -1],
    [1, -1, -3],
    [1, 1, 3],
    [1, 3, 1],
    [3, -1, -1],
    [3, 1, 1],
    [-3, -1, -1],
    [-3, 1, 1],
    [-1, -3, -1],
    [-1, -1, -3],
    [-1, 1, 3],
    [-1, 3, 1],
    [1, -3, 1],
    [1, -1, 3],
    [1, 1, -3],
    [1, 3, -1],
    [3, -1, 1],
    [3, 1, -1],
    [-4, 0, 0],
    [0, -4, 0],
    [0, 0, -4],
    [0, 0, 4],
    [0, 4, 0],
    [4, 0, 0],
    [-4, -2, -2],
    [-4, -2, 2],
    [-4, 2, -2],
    [-4, 2, 2],
    [-2, -4, -2],
    [-2, -4, 2],
    [-2, -2, -4],
    [-2, -2, 4],
    [-2, 2, -4],
    [-2, 2, 4],
    [-2, 4, -2],
    [-2, 4, 2],
    [2, -4, -2],
    [2, -4, 2],
    [2, -2, -4],
    [2, -2, 4],
    [2, 2, -4],
    [2, 2, 4],
    [2, 4, -2],
    [2, 4, 2],
    [4, -2, -2],
    [4, -2, 2],
    [4, 2, -2],
    [4, 2, 2],
    [-3, -3, -1],
    [-3, -1, -3],
    [-3, 1, 3],
    [-3, 3, 1],
    [-1, -3, -3],
    [-1, 3, 3],
    [1, -3, 3],
    [1, 3, -3],
    [3, -3, 1],
    [3, -1, 3],
    [3, 1, -3],
    [3, 3, -1],
    [-3, -3, 1],
    [-3, -1, 3],
    [-3, 1, -3],
    [-3, 3, -1],
    [-1, -3, 3],
    [-1, 3, -3],
    [1, -3, -3],
    [1, 3, 3],
    [3, -3, -1],
    [3, -1, -3],
    [3, 1, 3],
    [3, 3, 1],
    [-4, -4, 0],
    [-4, 0, -4],
    [-4, 0, 4],
    [-4, 4, 0],
    [0, -4, -4],
    [0, -4, 4],
    [0, 4, -4],
    [0, 4, 4],
    [4, -4, 0],
    [4, 0, -4],
    [4, 0, 4],
    [4, 4, 0],
    [-5, -1, -1],
    [-1, -5, -1],
    [-1, -1, -5],
    [-1, 1, 5],
    [-1, 5, 1],
    [1, -1, 5],
    [1, 1, -5],
    [5, -1, 1],
    [5, 1, -1],
    [-5, 1, 1],
    [-3, 3, -3],
    [1, -5, 1],
    [1, 5, -1],
    [3, -3, -3],
    [3, 3, 3],
    [-3, -3, 3],
    [-5, -1, 1],
    [-5, 1, -1],
    [-1, -1, 5],
    [-1, 1, -5],
    [1, -5, -1],
    [1, -1, -5],
    [1, 1, 5],
    [1, 5, 1],
    [5, 1, 1],
    [-3, -3, -3],
    [-3, 3, 3],
    [-1, -5, 1],
    [-1, 5, -1],
    [3, -3, 3],
    [5, -1, -1],
    [3, 3, -3],
    [-6, -2, 0],
    [-6, 0, -2],
    [-6, 0, 2],
    [-6, 2, 0],
    [-2, -6, 0],
    [-2, 0, -6],
    [-2, 0, 6],
    [-2, 6, 0],
    [0, -6, -2],
    [0, -6, 2],
    [0, -2, -6],
    [0, -2, 6],
    [0, 2, -6],
    [0, 2, 6],
    [0, 6, -2],
    [0, 6, 2],
    [2, -6, 0],
    [2, 0, -6],
    [2, 0, 6],
    [2, 6, 0],
    [6, -2, 0],
    [6, 0, -2],
    [6, 0, 2],
    [6, 2, 0],
    [-4, -4, -4],
    [-4, -4, 4],
    [-4, 4, -4],
    [-4, 4, 4],
    [4, -4, -4],
    [4, -4, 4],
    [4, 4, -4],
    [4, 4, 4],
    [-5, -3, 1],
    [-5, -1, 3],
    [-5, 1, -3],
    [-5, 3, -1],
    [-3, -5, 1],
    [-3, -1, 5],
    [-3, 1, -5],
    [-3, 5, -1],
    [-1, -5, 3],
    [-1, -3, 5],
    [-1, 3, -5],
    [-1, 5, -3],
    [1, -5, -3],
    [1, -3, -5],
    [1, 3, 5],
    [1, 5, 3],
    [3, -5, -1],
    [3, -1, -5],
    [3, 1, 5],
    [3, 5, 1],
    [5, -3, -1],
    [5, -1, -3],
    [5, 1, 3],
    [5, 3, 1],
    [-5, -3, -1],
    [-5, -1, -3],
    [-5, 1, 3],
    [-5, 3, 1],
    [-3, -5, -1],
    [-3, -1, -5],
    [-3, 1, 5],
    [-3, 5, 1],
    [-1, -5, -3],
    [-1, -3, -5],
    [-1, 3, 5],
    [-1, 5, 3],
    [1, -5, 3],
    [1, -3, 5],
    [1, 3, -5],
    [1, 5, -3],
    [3, -5, 1],
    [3, -1, 5],
    [3, 1, -5],
    [3, 5, -1],
    [5, -3, 1],
    [5, -1, 3],
    [5, 1, -3],
    [5, 3, -1],
    [-6, -4, -2],
    [-6, -4, 2],
    [-6, -2, -4],
    [-6, -2, 4],
    [-6, 2, -4],
    [-6, 2, 4],
    [-6, 4, -2],
    [-6, 4, 2],
    [-4, -6, -2],
    [-4, -6, 2],
    [-4, -2, -6],
    [-4, -2, 6],
    [-4, 2, -6],
    [-4, 2, 6],
    [-4, 6, -2],
    [-4, 6, 2],
    [-2, -6, -4],
    [-2, -6, 4],
    [-2, -4, -6],
    [-2, -4, 6],
    [-2, 4, -6],
    [-2, 4, 6],
    [-2, 6, -4],
    [-2, 6, 4],
    [2, -6, -4],
    [2, -6, 4],
    [2, -4, -6],
    [2, -4, 6],
    [2, 4, -6],
    [2, 4, 6],
    [2, 6, -4],
    [2, 6, 4],
    [4, -6, -2],
    [4, -6, 2],
    [4, -2, -6],
    [4, -2, 6],
    [4, 2, -6],
    [4, 2, 6],
    [4, 6, -2],
    [4, 6, 2],
    [6, -4, -2],
    [6, -4, 2],
    [6, -2, -4],
    [6, -2, 4],
    [6, 2, -4],
    [6, 2, 4],
    [6, 4, -2],
    [6, 4, 2],
    [-8, 0, 0],
    [0, -8, 0],
    [0, 0, -8],
    [0, 0, 8],
    [0, 8, 0],
    [8, 0, 0],
    [-5, -3, -3],
    [-5, 3, 3],
    [-3, -5, -3],
    [-3, -3, -5],
    [-3, 3, 5],
    [-3, 5, 3],
    [3, -5, 3],
    [3, -3, 5],
    [3, 3, -5],
    [3, 5, -3],
    [5, -3, 3],
    [5, 3, -3],
    [-5, -3, 3],
    [-5, 3, -3],
    [-3, -5, 3],
    [-3, -3, 5],
    [-3, 3, -5],
    [-3, 5, -3],
    [3, -5, -3],
    [3, -3, -5],
    [3, 3, 5],
    [3, 5, 3],
    [5, -3, -3],
    [5, 3, 3],
    [-7, -1, 1],
    [-7, 1, -1],
    [-5, -1, -5],
    [-1, -7, 1],
    [-1, -5, -5],
    [-1, -1, 7],
    [-1, 1, -7],
    [-1, 7, -1],
    [1, -7, -1],
    [1, -1, -7],
    [1, 1, 7],
    [1, 7, 1],
    [7, 1, 1],
    [-5, -5, -1],
    [-5, 1, 5],
    [-5, 5, 1],
    [-1, 5, 5],
    [1, -5, 5],
    [1, 5, -5],
    [5, -5, 1],
    [5, -1, 5],
    [5, 1, -5],
    [5, 5, -1],
    [7, -1, -1],
    [-7, -1, -1],
    [-1, -7, -1],
    [-1, -1, -7],
    [-1, 1, 7],
    [-1, 7, 1],
    [1, -7, 1],
    [1, -1, 7],
    [1, 1, -7],
    [1, 5, 5],
    [1, 7, -1],
    [5, 1, 5],
    [7, -1, 1],
    [7, 1, -1],
    [-7, 1, 1],
    [-5, -5, 1],
    [-5, -1, 5],
    [-5, 1, -5],
    [-5, 5, -1],
    [-1, -5, 5],
    [-1, 5, -5],
    [1, -5, -5],
    [5, -5, -1],
    [5, -1, -5],
    [5, 5, 1],
    [7, 3, 3],
    [-7, -3, 3],
    [-7, 3, -3],
    [-3, -7, 3],
    [-3, -3, 7],
    [-3, 7, -3],
    [-3, 3, -7],
    [3, 7, 3],
    [3, 3, 7],
    [7, -3, -3],
    [3, -7, -3],
    [3, -3, -7],
    [-7, -3, -3],
    [-7, 3, 3],
    [-3, -7, -3],
    [-3, -3, -7],
    [-3, 7, 3],
    [-3, 3, 7],
    [3, 3, -7],
    [3, 7, -3],
    [7, 3, -3],
    [7, -3, 3],
    [3, -7, 3],
    [3, -3, 7],
    [-7, -3, -1],
    [-7, -1, -3],
    [-7, 1, 3],
    [-7, 3, 1],
    [-5, -5, 3],
    [-5, -3, 5],
    [-5, 3, -5],
    [-5, 5, -3],
    [-3, -7, -1],
    [-3, -5, 5],
    [-3, -1, -7],
    [-3, 1, 7],
    [-3, 5, -5],
    [-3, 7, 1],
    [-1, -7, -3],
    [-1, -3, -7],
    [-1, 3, 7],
    [-1, 7, 3],
    [1, -7, 3],
    [1, -3, 7],
    [1, 3, -7],
    [1, 7, -3],
    [3, -7, 1],
    [3, -5, -5],
    [3, -1, 7],
    [3, 1, -7],
    [3, 5, 5],
    [3, 7, -1],
    [5, -5, -3],
    [5, -3, -5],
    [5, 3, 5],
    [5, 5, 3],
    [7, -3, 1],
    [7, -1, 3],
    [7, 1, -3],
    [7, 3, -1],
    [-7, -3, 1],
    [-7, -1, 3],
    [-7, 1, -3],
    [-7, 3, -1],
    [-5, -5, -3],
    [-5, -3, -5],
    [-5, 3, 5],
    [-5, 5, 3],
    [-3, -7, 1],
    [-3, -5, -5],
    [-3, -1, 7],
    [-3, 1, -7],
    [-3, 5, 5],
    [-3, 7, -1],
    [-1, -7, 3],
    [-1, -3, 7],
    [-1, 3, -7],
    [-1, 7, -3],
    [1, -7, -3],
    [1, -3, -7],
    [1, 3, 7],
    [1, 7, 3],
    [3, -7, -1],
    [3, -5, 5],
    [3, -1, -7],
    [3, 1, 7],
    [3, 5, -5],
    [3, 7, 1],
    [5, -5, 3],
    [5, -3, 5],
    [5, 3, -5],
    [5, 5, -3],
    [7, -3, -1],
    [7, -1, -3],
    [7, 1, 3],
    [7, 3, 1],
    [-2, 0, 0],
    [0, -2, 0],
    [0, 0, -2],
    [0, 0, 2],
    [0, 2, 0],
    [2, 0, 0],
    [-2, -2, -2],
    [-2, -2, 2],
    [-2, 2, -2],
    [-2, 2, 2],
    [2, -2, -2],
    [2, -2, 2],
    [2, 2, -2],
    [2, 2, 2],
    [-4, -2, 0],
    [-4, 0, -2],
    [-4, 0, 2],
    [-4, 2, 0],
    [-2, -4, 0],
    [-2, 0, -4],
    [-2, 0, 4],
    [-2, 4, 0],
    [0, -4, -2],
    [0, -4, 2],
    [0, -2, -4],
    [0, -2, 4],
    [0, 2, -4],
    [0, 2, 4],
    [0, 4, -2],
    [0, 4, 2],
    [2, -4, 0],
    [2, 0, -4],
    [2, 0, 4],
    [2, 4, 0],
    [4, -2, 0],
    [4, 0, -2],
    [4, 0, 2],
    [4, 2, 0],
    [-4, -2, -4],
    [-2, -4, -4],
    [2, 4, 4],
    [4, 2, 4],
    [-6, 0, 0],
    [-4, -4, -2],
    [-4, -4, 2],
    [-4, -2, 4],
    [-4, 2, -4],
    [-4, 2, 4],
    [-4, 4, -2],
    [-4, 4, 2],
    [-2, -4, 4],
    [-2, 4, -4],
    [-2, 4, 4],
    [0, -6, 0],
    [0, 0, -6],
    [0, 0, 6],
    [0, 6, 0],
    [2, -4, -4],
    [2, -4, 4],
    [2, 4, -4],
    [4, -4, -2],
    [4, -4, 2],
    [4, -2, -4],
    [4, -2, 4],
    [4, 2, -4],
    [4, 4, -2],
    [4, 4, 2],
    [6, 0, 0],
    [-6, -2, -2],
    [-6, -2, 2],
    [-6, 2, -2],
    [-6, 2, 2],
    [-2, -6, -2],
    [-2, -6, 2],
    [-2, -2, -6],
    [-2, -2, 6],
    [-2, 2, -6],
    [-2, 2, 6],
    [-2, 6, -2],
    [-2, 6, 2],
    [2, -6, -2],
    [2, -6, 2],
    [2, -2, -6],
    [2, -2, 6],
    [2, 2, -6],
    [2, 2, 6],
    [2, 6, -2],
    [2, 6, 2],
    [6, -2, -2],
    [6, -2, 2],
    [6, 2, -2],
    [6, 2, 2],
    [-6, -4, 0],
    [-6, 0, -4],
    [-4, -6, 0],
    [-4, 0, -6],
    [-4, 0, 6],
    [-4, 6, 0],
    [0, -6, -4],
    [0, -4, -6],
    [0, -4, 6],
    [0, 4, -6],
    [0, 4, 6],
    [0, 6, 4],
    [4, -6, 0],
    [4, 0, -6],
    [4, 0, 6],
    [4, 6, 0],
    [6, 0, 4],
    [6, 4, 0],
    [-6, 0, 4],
    [-6, 4, 0],
    [0, -6, 4],
    [0, 6, -4],
    [6, -4, 0],
    [6, 0, -4],
]

axesactive = (
    [-4, 2, 2],
    [2, -4, 2],
    [0, -4, 0],
    [0, 4, 0],
    [4, 0, 0],
    [-4, 0, 0],
    [0, 2, 2],
    [2, 0, 2],
    [-2, 2, 0],
    [0, -2, 2],
    [-2, 0, 2],
    [2, -2, 0],
    [1, -1, 1],
    [-1, 1, 1],
)


def getpoints(axis, testpoints, E, a, N):
    normaxis = np.linalg.norm(axis)
    return np.where(
        ne.evaluate(
            "abs(tp1*a1+tp2*a2+tp3*a3-c)<0.001",
            {
                "c": (12398 * normaxis ** 2 / (E * 2 * a)),
                "tp1": testpoints[:, 0],
                "tp2": testpoints[:, 1],
                "tp3": testpoints[:, 2],
                "a1": axis[0],
                "a2": axis[1],
                "a3": axis[2],
            },
        ).reshape(N)
    )


def residual(params, testpointsdet, axes):
    r = Rotation.from_euler("xyz", np.array((params["rot_x"].value, params["rot_y"].value, params["rot_z"].value)))
    testpoints = r.apply(testpointsdet + np.array((params["trans_x"], params["trans_y"], params["trans_z"])), inverse=False)
    testpoints = testpoints / np.linalg.norm(testpoints, axis=1)[:, None]
    res = (np.dot(testpoints, axes.T)) / (np.linalg.norm(axes, axis=1)) ** 2 - params["c"].value
    res = np.take_along_axis(res, np.nanargmin(np.abs(res), axis=1)[:, None], 1)
    return res


class Kossel(QMainWindow, kosselui.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.N0 = 1000
        self.N1 = 1000

        self.setupUi(self)
        self.plotButton.clicked.connect(self.plot_data)
        self.clearButton.clicked.connect(self.clear_data)
        self.loadButton.clicked.connect(self.load_file)
        self.peakButton.clicked.connect(self.find_peaks)
        self.fitButton.clicked.connect(self.fit_data)
        self.datasetCombo.currentTextChanged.connect(self.plot_bg)
        self.removeBackgroundBox.stateChanged.connect(self.plot_bg)
        self.data = np.zeros((self.N0, self.N1))
        self.data[0, 0] = 1
        self.data[-1, -1] = 1
        self.inputfile = None
        self.bgplot = self.plotarea.canvas.ax.matshow(self.data, vmin=0, vmax=1)
        self.peaks = np.zeros((self.N0, self.N1))
        self.peaks[:] = np.nan
        self.peaksplot = self.plotarea.canvas.ax.matshow(self.peaks, vmin=0, vmax=1, cmap="gray")
        self.plotarea.canvas.ax.set_xlim(0, self.N1)
        self.plotarea.canvas.ax.set_ylim(self.N0, 0)
        self.plotarea.canvas.draw_idle()
        self.rangeSlider.startValueChanged.connect(self.setclim)
        self.rangeSlider.endValueChanged.connect(self.setclim)
        self.redrawtimer = QTimer()
        self.redrawtimer.setSingleShot(True)
        self.redrawtimer.timeout.connect(self.plotarea.canvas.draw)
        for i, ax in enumerate(axes):
            label = np.array2string(np.array(ax), precision=0)
            item = QListWidgetItem(label)
            item.setData(1, ax)
            self.reflexList.addItem(item)
            if ax in axesactive:
                item.setSelected(True)
        for el in [
            self.angleXSpin,
            self.angleYSpin,
            self.angleZSpin,
            self.transXSpin,
            self.transYSpin,
            self.transZSpin,
            self.energySpin,
            self.latticeSpin,
        ]:
            el.valueChanged.connect(self.invalidate_plot)

        self.plotpoints = {}
        self.testpoints = None

        def styles(i=0):
            colors = (
                "#1f77b4",
                "#aec7e8",
                "#ff7f0e",
                "#ffbb78",
                "#2ca02c",
                "#98df8a",
                "#d62728",
                "#ff9896",
                "#9467bd",
                "#c5b0d5",
                "#8c564b",
                "#c49c94",
                "#e377c2",
                "#f7b6d2",
                "#7f7f7f",
                "#c7c7c7",
                "#bcbd22",
                "#dbdb8d",
                "#17becf",
                "#9edae5",
            )
            markers = (".", "+", "x")
            while True:
                yield colors[i % len(colors)], markers[(i // len(colors)) % len(markers)]
                i += 1

        self.sgen = styles()
        self.inputfilename = None

        self.plotarea.canvas.mpl_connect("motion_notify_event", self.mpl_move)
        self.plotarea.canvas.mpl_connect("button_release_event", self.mpl_release)

    def mpl_move(self, event):
        if event.button == 1:
            tmp = self.data[int(event.ydata) - 1 : int(event.ydata) + 2, int(event.xdata) - 1 : int(event.xdata) + 2]
            m = tmp == tmp.max()
            self.peaks[int(event.ydata) - 1 : int(event.ydata) + 2, int(event.xdata) - 1 : int(event.xdata) + 2][m] = 1
        elif event.button == 3:
            self.peaks[int(event.ydata) - 5 : int(event.ydata) + 5, int(event.xdata) - 5 : int(event.xdata) + 5] = np.nan

    def mpl_release(self, event):
        self.peaksplot.set_array(self.peaks)
        self.plotarea.canvas.draw_idle()

    def plot_bg(self):
        if self.inputfile is not None:
            data = np.array(self.inputfile[self.datasetCombo.currentText()])
            if data.ndim == 2:
                if self.removeBackgroundBox.isChecked():
                    data = data - snd.grey_opening(data, structure=skm.disk(10))
                data = data - np.nanmin(data)
                data = data / np.nanmax(data)
                self.data = data
                self.bgplot.remove()
                self.bgplot = self.plotarea.canvas.ax.matshow(self.data, vmin=0, vmax=1, zorder=0)
                if data.shape[0] != self.N0 or data.shape[1] != self.N1:
                    self.N0 = data.shape[0]
                    self.N1 = data.shape[1]
                    self.peaks = np.zeros((self.N0, self.N1))
                    self.peaks[:] = np.nan
                    self.peaksplot = self.plotarea.canvas.ax.matshow(self.peaks, vmin=0, vmax=1, cmap="gray", zorder=1)
                    self.invalidate_plot()
                    self.plotarea.canvas.draw_idle()

                self.plotarea.canvas.ax.set_xlim(0, self.N1)
                self.plotarea.canvas.ax.set_ylim(self.N0, 0)

        self.plotarea.canvas.draw_idle()

    def setclim(self):
        clim = np.array(self.rangeSlider.getRange()) / 100
        self.bgplot.set_clim(clim)
        self.redrawtimer.start(100)
        # self.plotarea.canvas.draw_idle()

    def invalidate_plot(self):
        for label in self.plotpoints:
            self.plotpoints[label][0] = None
        self.testpoints = None

    def load_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", ".", "h5 (*.h5 *.hdf5)")
        self.inputfileLabel.setText(os.path.split(fname[0])[1])
        self.inputfilename = fname[0]
        try:
            if self.inputfile is not None:
                self.inputfile.close()
                self.inputfile = None
            inputfile = h5py.File(fname[0], "r")
            self.datasetCombo.clear()
            self.datasetCombo.addItems(inputfile.keys())
            self.inputfile = inputfile
        except Exception as e:
            print("error opening file:", e)

    def fit_data(self):
        x, y = np.where(~np.isnan(self.peaks))
        testpointsdet = np.array((x, y, [0] * len(x))).T - np.array((self.N0 // 2, self.N1 // 2, 0))
        fit_params = lmfit.Parameters()
        fit_params.add(
            "rot_x",
            value=self.angleXSpin.value() / 180 * np.pi,
            min=(self.angleXSpin.value() - 15) / 180 * np.pi,
            max=(self.angleXSpin.value() + 15) / 180 * np.pi,
            vary=not self.rotXFixBox.isChecked(),
        )
        fit_params.add(
            "rot_y",
            value=self.angleYSpin.value() / 180 * np.pi,
            min=(self.angleYSpin.value() - 15) / 180 * np.pi,
            max=(self.angleYSpin.value() + 15) / 180 * np.pi,
            vary=not self.rotYFixBox.isChecked(),
        )
        fit_params.add(
            "rot_z",
            value=self.angleZSpin.value() / 180 * np.pi,
            min=(self.angleZSpin.value() - 15) / 180 * np.pi,
            max=(self.angleZSpin.value() + 15) / 180 * np.pi,
            vary=not self.rotZFixBox.isChecked(),
        )
        fit_params.add(
            "trans_x",
            value=self.transXSpin.value(),
            min=self.transXSpin.value() - 100,
            max=self.transXSpin.value() + 100,
            vary=not self.transXFixBox.isChecked(),
        )
        fit_params.add(
            "trans_y",
            value=self.transYSpin.value(),
            min=self.transYSpin.value() - 100,
            max=self.transYSpin.value() + 100,
            vary=not self.transYFixBox.isChecked(),
        )
        fit_params.add(
            "trans_z",
            value=self.transZSpin.value(),
            min=self.transZSpin.value() - 100,
            max=self.transZSpin.value() + 100,
            vary=not self.transZFixBox.isChecked(),
        )
        c = 12.398 / (self.energySpin.value() * 2 * self.latticeSpin.value())
        fit_params.add("c", value=c, min=0.8 * c, max=1.2 * c, vary=not self.latticeFixBox.isChecked())
        axs = np.array([np.array(item.data(1)) for item in self.reflexList.selectedItems()])
        minner = lmfit.Minimizer(residual, fit_params, fcn_args=(testpointsdet, axs))
        result = minner.minimize(method="bfgs")
        print(lmfit.fit_report(result))
        self.angleXSpin.setValue(result.params["rot_x"] * 180 / np.pi)
        self.angleYSpin.setValue(result.params["rot_y"] * 180 / np.pi)
        self.angleZSpin.setValue(result.params["rot_z"] * 180 / np.pi)
        self.transXSpin.setValue(result.params["trans_x"])
        self.transYSpin.setValue(result.params["trans_y"])
        self.transZSpin.setValue(result.params["trans_z"])
        self.latticeSpin.setValue(12.398 / (self.energySpin.value() * 2 * result.params["c"]))

    def plot_data(self):
        if self.testpoints is None:
            self.clear_data()
            r = Rotation.from_euler("xyz", np.array((self.angleXSpin.value(), self.angleYSpin.value(), self.angleZSpin.value())) / 180 * np.pi)
            Y, X, Z = np.meshgrid(np.arange(-self.N1 // 2, self.N1 // 2, 1), np.arange(-self.N0 // 2, self.N0 // 2, 1), 0)
            testpoints = np.array([m.ravel() for m in [X, Y, Z]]).T
            testpoints = r.apply(testpoints + np.array((self.transXSpin.value(), self.transYSpin.value(), self.transZSpin.value())), inverse=False)
            self.testpoints = testpoints / np.linalg.norm(testpoints, axis=1)[:, None]

        E = self.energySpin.value() * 1000
        a = self.latticeSpin.value()

        items = self.reflexList.selectedItems()
        selectedlabels = [item.data(0) for item in items]
        for label, item in self.plotpoints.items():
            if label not in selectedlabels:
                if item[1] is not None:
                    item[1].remove()
                    item[1] = None
                if item[2] is not None:
                    item[2].remove()
                    item[2] = None

        for k, item in enumerate(items):
            ax = np.array(item.data(1), dtype=int)
            label = np.array2string(ax, precision=0)
            ax = ax.astype(float)
            self.progressBar.setValue(int(k / len(items) * 100))
            self.progressBar.update()
            QApplication.processEvents()
            if label in self.plotpoints and self.plotpoints[label][0] is not None:
                continue
            points = getpoints(ax, testpoints=self.testpoints, E=E, a=a, N=(self.N0, self.N1))
            self.plotpoints[label] = [points, None, None]

        for label in selectedlabels:
            points = self.plotpoints[label][0]

            if len(points[0]) > 0:
                s = next(self.sgen)
                if self.plotpoints[label][1] is None:
                    self.plotpoints[label][1] = self.plotarea.canvas.ax.scatter(points[1], points[0], label=label, c=s[0], s=1)

                if self.plotpoints[label][2] is None:
                    for j in range(15):
                        i = np.random.choice(np.arange(0, len(points[0])))
                        if 10 < points[0][i] < (self.N0 - 20) and 10 < points[1][i] < (self.N1 - 100):
                            self.plotpoints[label][2] = self.plotarea.canvas.ax.text(points[1][i] + 5, points[0][i] + 5, s=label, c=s[0])
                            break
                        else:
                            self.plotpoints[label][2] = self.plotarea.canvas.ax.text(
                                np.clip(points[1][i] + 5, 20, self.N1 - 20), np.clip(points[0][i] + 5, 20, self.N0 - 100), s=label, c=s[0]
                            )

        self.plotarea.canvas.ax.set_xlim(0, self.N1)
        self.plotarea.canvas.ax.set_ylim(self.N0, 0)
        self.plotarea.canvas.draw_idle()

    def clear_data(self):
        for label in self.plotpoints:
            if self.plotpoints[label][1] is not None:
                self.plotpoints[label][1].remove()
                self.plotpoints[label][1] = None

            if self.plotpoints[label][2] is not None:
                self.plotpoints[label][2].remove()
                self.plotpoints[label][2] = None

        self.plotarea.canvas.draw_idle()

    def find_peaks(self):
        if self.data is not None:
            self.peaks[:] = np.nan
            self.peaks[self.data > (self.thresholdSlider.value() / 100)] = 1
            self.peaksplot.set_array(self.peaks)
            self.plotarea.canvas.draw_idle()


def sigint_handler(*args):
    sys.stderr.write("\r")
    QApplication.quit()


if __name__ == "__main__":
    print('starting kossel')
    signal.signal(signal.SIGINT, sigint_handler)
    app = QApplication(sys.argv)
    timer = QTimer()
    timer.start(250)
    timer.timeout.connect(lambda: None)
    form = Kossel()
    form.show()
    r = app.exec_()
    sys.exit(r)
