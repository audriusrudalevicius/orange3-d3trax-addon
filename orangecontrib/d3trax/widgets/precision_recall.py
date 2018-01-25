from collections import namedtuple

import numpy as np
import sklearn.metrics as skl_metrics

from AnyQt import QtWidgets
from AnyQt.QtGui import QColor, QPen, QPalette, QFont
from AnyQt.QtCore import Qt

import pyqtgraph as pg

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.evaluate.utils import check_results_adequacy
from Orange.widgets.utils import colorpalette, colorbrewer
from Orange.widgets.widget import Input
from Orange.canvas import report

CurvePoints = namedtuple(
    "CurvePoints",
    ["cases", "tpr", "thresholds"]
)
CurvePoints.is_valid = property(lambda self: self.cases.size > 0)

PlotCurve = namedtuple(
    "PlotCurve",
    ["curve",
     "curve_item", "curve_label"]
)

LiftCurve = namedtuple(
    "LiftCurve",
    ["points", "avg"],
)
LiftCurve.is_valid = property(lambda self: self.points.is_valid)


class OWPrecisionRecall(widget.OWWidget):
    name = "Precision vs Recall"
    description = "The precision-recall curve shows the tradeoff between precision and recall for different threshold."
    priority = 1020
    category = "Evaluate"

    class Inputs:
        evaluation_results = Input("Evaluation Results", Orange.evaluation.Results)

    class Warning(widget.OWWidget.Warning):
        empty_results = widget.Msg(
            "Empty results on input. There is nothing to display.")

    target_index = settings.Setting(0)
    selected_classifiers = settings.Setting([])
    show_average_precisions = settings.Setting(False)

    graph_name = "plot"

    def __init__(self):
        super().__init__()

        self.classifier_names = []
        self.results = None
        self.colors = []
        self._curve_data = {}

        box = gui.vBox(self.controlArea, "Plot")
        tbox = gui.vBox(box, "Target Class")
        tbox.setFlat(True)

        self.target_cb = gui.comboBox(
            tbox, self, "target_index", callback=self._on_target_changed,
            contentsLength=8)

        cbox = gui.vBox(box, "Classifiers")
        cbox.setFlat(True)
        self.classifiers_list_box = gui.listBox(
            cbox, self, "selected_classifiers", "classifier_names",
            selectionMode=QtWidgets.QListView.MultiSelection,
            callback=self._on_classifiers_changed)
        self.show_average_precisions_cb = gui.checkBox(
            cbox, self, "show_average_precisions", "Show average precisions",
            callback=self._replot
        )
        self.plotview = pg.GraphicsView(background="w")

        self.plot = pg.PlotItem(enableMenu=False)
        self.plot.hideButtons()

        pen = QPen(self.palette().color(QPalette.Text))

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        axis = self.plot.getAxis("bottom")
        axis.setTickFont(tickfont)
        axis.setPen(pen)
        axis.setLabel("Recall")

        axis = self.plot.getAxis("left")
        axis.setTickFont(tickfont)
        axis.setPen(pen)
        axis.setLabel("Precision")

        self.plot.showGrid(True, True, alpha=0.1)
        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.05), padding=0.05)

        self.plotview.setCentralItem(self.plot)
        self.mainArea.layout().addWidget(self.plotview)

    @Inputs.evaluation_results
    def set_results(self, results):
        self.clear()
        self.results = check_results_adequacy(results, self.Error)
        if self.results is not None:
            self._initialize(results)
            self._setup_plot()

    def clear(self):
        self.plot.clear()
        self.colors = []
        self.results = None
        self._curve_data = {}
        self.classifier_names = []
        self.target_index = 0
        self.target_cb.clear()

    def onDeleteWidget(self):
        self.clear()

    def send_report(self):
        if self.results is None:
            return
        classifiers = self.selected_classifiers
        results = [
            pvr_curve_from_results(self.results, self.target_index, clf_idx) for clf_idx in classifiers
        ]
        averagePrecisions = pg.OrderedDict(
            [(self.classifier_names[clf_idx], "{:.3f}".format(results[clf_idx][3])) for clf_idx in classifiers]
        )
        caption = report.list_legend(self.classifiers_list_box,
                                     self.selected_classifiers)
        self.report_items((
            ("Target class", self.target_cb.currentText()),
            ("Average precisions", tuple(averagePrecisions.items()))
        ))

        self.report_plot()
        self.report_caption(caption)


    def _initialize(self, results):
        N = len(results.predicted)

        names = getattr(results, "learner_names", None)
        if names is None:
            names = ["#{}".format(i + 1) for i in range(N)]

        scheme = colorbrewer.colorSchemes["qualitative"]["Dark2"]
        if N > len(scheme):
            scheme = colorpalette.DefaultRGBColors
        self.colors = colorpalette.ColorPaletteGenerator(N, scheme)

        self.classifier_names = names
        self.selected_classifiers = list(range(N))
        for i in range(N):
            item = self.classifiers_list_box.item(i)
            item.setIcon(colorpalette.ColorPixmap(self.colors[i]))

        self.target_cb.addItems(results.data.domain.class_var.values)

    def _on_target_changed(self):
        self._replot()

    def _on_classifiers_changed(self):
        self._replot()

    def plot_curves(self, target, clf_idx):
        if (target, clf_idx) not in self._curve_data:
            curve = pvrCurve_from_results(self.results, clf_idx, target)
            color = self.colors[clf_idx]
            pen = QPen(color, 1)
            pen.setCosmetic(True)
            shadow_pen = QPen(pen.color().lighter(160), 2.5)
            shadow_pen.setCosmetic(True)
            item = pg.PlotDataItem(
                curve.points[0], curve.points[1],
                pen=pen, shadowPen=shadow_pen,
                symbol="+", symbolSize=3, symbolPen=shadow_pen,
                antialias=True
            )
            item_label = pg.TextItem(
                text="AP{:.3f}".format(curve[1]),
                color=color
            )
            item_label.setPos(0.1, 1 - (clf_idx / 10))

            self._curve_data[target, clf_idx] = \
                PlotCurve(curve, item, item_label)

        return self._curve_data[target, clf_idx]

    def _setup_plot(self):
        target = self.target_index
        selected = self.selected_classifiers
        curves = [self.plot_curves(target, clf_idx) for clf_idx in selected]
        for curve in curves:
            self.plot.addItem(curve.curve_item)
            if self.show_average_precisions:
                self.plot.addItem(curve.curve_label)

        warning = ""
        if not all(c.curve.is_valid for c in curves):
            if any(c.curve.is_valid for c in curves):
                warning = "Some curves are undefined"
            else:
                warning = "All curves are undefined"

        self.warning(warning)

    def _replot(self):
        self.plot.clear()
        if self.results is not None:
            self._setup_plot()


def pvr_curve(ytrue, ypred, target=1):
    P = np.sum(ytrue == target)
    N = ytrue.size - P

    if P == 0 or N == 0:
        # Undefined TP and FP rate
        return np.array([]), np.array([]), np.array([]), .0

    precision, recall, thresholds = skl_metrics.precision_recall_curve(ytrue, ypred)
    average_precision = skl_metrics.average_precision_score(ytrue, ypred)

    return precision, recall, thresholds, average_precision


def pvr_curve_from_results(results, target, clf_idx, fold=slice(0, -1)):
    actual = results.actual[fold]
    y_scores = results.probabilities[clf_idx][fold][:, target]

    precision, recall, thresholds, average_precision = pvr_curve(actual, y_scores, target)
    return precision, recall, thresholds, average_precision


def pvrCurve_from_results(results, clf_index, target):
    x, y, thresholds, avg = pvr_curve_from_results(results, target, clf_index)

    points = CurvePoints(y, x, thresholds)
    return LiftCurve(points, avg)


def main():
    import sip
    from AnyQt.QtWidgets import QApplication
    from Orange.classification import (LogisticRegressionLearner, SVMLearner,
                                       NuSVMLearner)

    app = QApplication([])
    w = OWPrecisionRecall()
    w.show()
    w.raise_()

    data = Orange.data.Table("ionosphere")
    results = Orange.evaluation.CrossValidation(
        data,
        [
            LogisticRegressionLearner(penalty="l2"),
            LogisticRegressionLearner(penalty="l1"),
            SVMLearner(probability=True),
            NuSVMLearner(probability=True)
        ],
        store_data=True
    )
    results.learner_names = [
        "LR l2",
        "LR l1",
        "SVM",
        "Nu SVM"
    ]
    w.set_results(results)
    rval = app.exec_()

    sip.delete(w)
    del w
    app.processEvents()
    del app
    return rval


if __name__ == "__main__":
    main()
