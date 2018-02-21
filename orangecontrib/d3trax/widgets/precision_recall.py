from collections import namedtuple

import numpy as np
import sklearn.metrics as skl_metrics

from AnyQt import QtWidgets
from AnyQt.QtGui import QPen, QPalette, QFont

import pyqtgraph as pg

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.evaluate.utils import check_results_adequacy
from Orange.widgets.utils import colorpalette, colorbrewer
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME, create_annotated_table
from Orange.widgets.widget import Input, Output
from Orange.canvas import report
from Orange.data import Table

CurvePoints = namedtuple(
    "CurvePoints",
    ["cases", "tpr", "thresholds"]
)
CurvePoints.is_valid = property(lambda self: self.cases.size > 0)

PlotCurve = namedtuple(
    "PlotCurve",
    ["curve", "curve_item", "curve_label"]
)

LiftCurve = namedtuple(
    "LiftCurve",
    ["points", "avg", "cut_off"],
)
LiftCurve.is_valid = property(lambda self: self.points.is_valid)


def check_results(results, error_group, check_nan=True):
    results = check_results_adequacy(results, error_group, check_nan)
    if results is not None and len(results.data.domain.class_var.values) > 2:
        error_group.invalid_results("Multiclass not supported.")
        return None
    return results


class OWPrecisionRecall(widget.OWWidget):
    name = "Precision vs Recall"
    description = "The precision-recall curve shows the tradeoff between precision and recall for different threshold."
    priority = 1020
    category = "Evaluate"

    class Inputs:
        evaluation_results = Input("Evaluation Results", Orange.evaluation.Results)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True, id="selected-data")
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)

    class Warning(widget.OWWidget.Warning):
        empty_results = widget.Msg(
            "Empty results on input. There is nothing to display.")

    target_index = settings.Setting(0)
    selected_classifiers = settings.Setting([])
    show_average_precisions = settings.Setting(False)
    use_cut_off_point = settings.Setting(False)
    cut_off_point = settings.Setting(0.5)
    autocommit = settings.Setting(True)

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
            selectionMode=QtWidgets.QListView.SingleSelection,
            callback=self._on_classifiers_changed)
        self.show_average_precisions_cb = gui.checkBox(
            cbox, self, "show_average_precisions", "Show average precisions",
            callback=lambda: self._replot()
        )
        self.cut_off_point_box = gui.spin(
            cbox, self, "cut_off_point", 0, 1.0,
            step=0.005, spinType=float,
            label="Cut-off point",
            checked="use_cut_off_point",
            checkCallback=self._on_cut_off_changed,
            callback=self._on_cut_off_changed
        )
        self.cut_off_precision_label = gui.label(cbox, self, "")

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

        vbox = gui.vBox(self.controlArea, "Output")
        self.output_items_label = gui.label(vbox, self, "")
        gui.auto_commit(vbox, self, "autocommit", "Send Selection", "Send Automatically", box=False)

        self.plotview.setCentralItem(self.plot)
        self.mainArea.layout().addWidget(self.plotview)

    @Inputs.evaluation_results
    def set_results(self, results):
        self.clear()
        self.results = check_results(results, self.Error)
        if self.results is not None:
            self._initialize(results)
            self._setup_plot()
            if self.autocommit:
                self.commit()

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
        pvr_stats = [
            pvr_curve_from_results(self.results, self.target_index, clf_idx) for clf_idx in classifiers
        ]
        averagePrecisions = pg.OrderedDict(
            [(self.classifier_names[clf_idx], "{:.3f}".format(pvr_stats[clf_idx][3])) for clf_idx in classifiers]
        )
        caption = report.list_legend(self.classifiers_list_box, self.selected_classifiers)
        tn = fp = fn = tp = tpr = ppv = None
        if self.use_cut_off_point and self.cut_off_point >= 0 and len(classifiers) > 0:
            clf_idx = (classifiers[:1] or [None])[0]
            matrix, tpr, ppv = cut_off_from_results(self.results, clf_idx, self.cut_off_point)
            tn, fp, fn, tp = matrix
        self.report_items((
            ("Target class", self.target_cb.currentText()),
            ("Average precisions", tuple(averagePrecisions.items())),
            (
                "Cutoff",
                "{:.3f}".format(self.cut_off_point) if self.cut_off_point > 0 and self.use_cut_off_point else None
            ),
            ("Recall", "{:.3f}".format(tpr) if tpr else None),
            ("Precision", "{:.3f}".format(ppv) if ppv else None),
            ("TN", tn if tn else None),
            ("FP", fp if fp else None),
            ("TP", tp if tp else None),
            ("FN", fn if fn else None),
        ))

        self.report_plot()
        self.report_caption(caption)

    def commit(self):
        self.send_selection()

    def send_selection(self):
        if self.results is None or len(self.selected_classifiers) < 1 or len(self.classifier_names) < 1:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            self.output_items_label.setText("Outputting no items")
            return

        target = self.target_index
        clf_name = self.classifier_names[self.selected_classifiers[0]]
        class_var = self.results.data.domain.class_var
        extra = []
        mask = []
        metas = self.results.data.domain.metas
        predicted = self.results.predicted[self.selected_classifiers[0]]
        probs = self.results.probabilities[self.selected_classifiers[0]]

        if self.cut_off_point > 0 and self.use_cut_off_point:
            mask = probs[:, target] >= self.cut_off_point

        extra.append(predicted.reshape(-1, 1))
        var = Orange.data.DiscreteVariable(
            "{}({})".format(class_var.name, clf_name),
            class_var.values
        )
        metas = metas + (var,)

        extra.append(np.array(probs, dtype=object))
        pvars = [Orange.data.ContinuousVariable("p({})".format(value))
                 for value in class_var.values]
        metas = metas + tuple(pvars)

        domain = Orange.data.Domain(self.results.data.domain.attributes,
                                    self.results.data.domain.class_vars,
                                    metas)
        data = self.results.data.transform(domain)
        if len(extra):
            data.metas[:, len(self.results.data.domain.metas):] = \
                np.hstack(tuple(extra))

        total = len(probs[:, target])
        data.name = clf_name

        if len(mask) > 0:
            annotated_data = create_annotated_table(data, mask)
            data = data[mask]
        else:
            annotated_data = create_annotated_table(data, [])
            data = None

        self.output_items_label.setText("Outputting {:d} of {:d} items".format(len(data) if data else 0, total))

        self.Outputs.selected_data.send(data)
        self.Outputs.annotated_data.send(annotated_data)

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
        self.selected_classifiers = [0] if N >= 0 else []
        for i in range(N):
            item = self.classifiers_list_box.item(i)
            item.setIcon(colorpalette.ColorPixmap(self.colors[i]))

        self.target_cb.addItems(results.data.domain.class_var.values)

    def _on_cut_off_changed(self):
        self._replot()
        self.send_selection()

    def _on_target_changed(self):
        self._replot()
        self.send_selection()

    def _on_classifiers_changed(self):
        self._replot()
        self.send_selection()

    def plot_curves(self, target, clf_idx):
        if (target, clf_idx) not in self._curve_data:
            curve = pvrCurve_from_results(self.results, clf_idx)
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
                text="AP{:.3f}".format(curve.avg),
                color=color
            )
            item_label.setPos(0.1, 1 - (clf_idx / 10))

            self._curve_data[target, clf_idx] = PlotCurve(curve, item, item_label)

        return self._curve_data[target, clf_idx]

    def _setup_plot(self):
        target = self.target_index
        selected = self.selected_classifiers
        curves = [self.plot_curves(target, clf_idx) for clf_idx in selected]
        for curve in curves:
            self.plot.addItem(curve.curve_item)
            if self.show_average_precisions:
                self.plot.addItem(curve.curve_label)

        if self.use_cut_off_point and self.cut_off_point >= 0:
            for clf_idx in selected:
                matrix, tpr, ppv = cut_off_from_results(self.results, clf_idx, self.cut_off_point, target)
                tn, fp, fn, tp = matrix
                color = self.colors[clf_idx]
                pen = QPen(color, 1)
                pen.setCosmetic(True)
                line = self.plot.addLine(x=tpr)
                line.setPen(pen)
                self.cut_off_precision_label.setText(
                    "TPR: {:.3f} PPV: {:.3f}\nTP:{:d} FN:{:d}\nFP:{:d} TN:{:d}".format(
                        tpr, ppv, tp, fn, fp, tn
                    ))
                line = self.plot.addLine(y=ppv)
                line.setPen(pen)
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

    precision, recall, thresholds = skl_metrics.precision_recall_curve(ytrue.ravel(), ypred.ravel())
    average_precision = skl_metrics.average_precision_score(ytrue, ypred, average="micro")

    fpr, tpr, threshold = skl_metrics.roc_curve(ytrue, ypred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return precision, recall, thresholds, average_precision, optimal_threshold


def pvr_curve_from_results(results, target, clf_idx):
    actual = results.actual
    y_scores = results.probabilities[clf_idx][:, target]

    return pvr_curve(actual, y_scores, target)


def reduced_results(results, clf_idx, cut_off, target=1):
    y_scores = results.probabilities[clf_idx][:, target]
    mask = np.where(y_scores >= cut_off)[0]
    actual_masked = results.actual[mask]
    predicted_masked = results.predicted[clf_idx][mask]
    probs_masked = results.probabilities[clf_idx][mask]
    data = results.data[mask]

    return actual_masked, predicted_masked, probs_masked, data


def cut_off_from_results(results, clf_idx, cut_off, target=1):
    y_scores = results.probabilities[clf_idx][:, target]
    cut_off_mask = y_scores >= cut_off

    actual_masked, predicted_masked, probs_masked, data = reduced_results(results, clf_idx, cut_off, target)
    labels = np.arange(len(results.domain.class_var.values))

    if not results.actual.size:
        # scikit-learn will not return an zero matrix
        return np.zeros((len(labels), len(labels)))

    matrix = skl_metrics.confusion_matrix(data.Y, predicted_masked, labels).ravel()
    matrix_plot = skl_metrics.confusion_matrix(results.actual, cut_off_mask, labels).ravel()

    tn, fp, fn, tp = matrix_plot
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    return matrix_plot, tpr, ppv


def pvrCurve_from_results(results, clf_index, target=1):
    x, y, thresholds, avg, cut_off = pvr_curve_from_results(results, target, clf_index)

    points = CurvePoints(y, x, thresholds)
    return LiftCurve(points, avg, cut_off)


def table_concat(tables):
    """
    Concatenate a list of tables.

    The resulting table will have a union of all attributes of `tables`.

    """
    attributes = []
    class_vars = []
    metas = []
    variables_seen = set()

    for table in tables:
        attributes.extend(v for v in table.domain.attributes if v not in variables_seen)
        variables_seen.update(table.domain.attributes)
        class_vars.extend(v for v in table.domain.class_vars if v not in variables_seen)
        variables_seen.update(table.domain.class_vars)
        metas.extend(v for v in table.domain.metas if v not in variables_seen)
        variables_seen.update(table.domain.metas)

    domain = Orange.data.Domain(attributes, class_vars, metas)
    tables = [tab.transform(domain) for tab in tables]

    return tables[0].concatenate(tables, axis=0)


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
