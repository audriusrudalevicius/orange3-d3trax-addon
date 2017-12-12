import tempfile

from AnyQt.QtCore import Qt

from Orange.base import Learner
from Orange.data import Table
from Orange.modelling import SklTreeLearner

from orangecontrib.d3trax.modeling import CatBoostLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Msg, Input


class OWCatBoost(OWBaseLearner):
    name = "CatBoost"
    description = "An ensemble meta-algorithm"
    icon = "icons/AdaBoost.svg"
    category = "Model"

    priority = 80

    LEARNER = CatBoostLearner
    DEFAULT_BASE_ESTIMATOR = None

    random_seed = Setting(0)
    n_iterations = Setting(50)
    depth = Setting(6)
    learning_rate = Setting(.03)
    loss_index = Setting(0)
    use_random_seed = Setting(False)
    train_dir = Setting(tempfile.gettempdir())
    losses = ["Logloss", "MultiClass"]

    class Error(OWBaseLearner.Error):
        no_weight_support = Msg('The base learner does not support weights.')

    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", Learner)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        self.base_estimator = self.DEFAULT_BASE_ESTIMATOR

        self.base_label = gui.label(
            box, self, "Base learner: " + self.base_estimator.name.title() if self.base_estimator is not None  else 'None')
        self.path_edit = gui.lineEdit(box, self, "train_dir", label="Path for training:",
          alignment=Qt.AlignRight, callback=self.settings_changed)
        self.n_estimators_spin = gui.spin(
            box, self, "n_iterations", 1, 100, label="Number of estimators:",
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        self.depth_spin = gui.spin(
            box, self, "depth", 1, 100, label="Depth:",
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        self.learning_rate_spin = gui.doubleSpin(
            box, self, "learning_rate", 1e-5, 1.0, 1e-5,
            label="Learning rate:", decimals=5, alignment=Qt.AlignRight,
            controlWidth=80, callback=self.settings_changed)
        self.random_seed_spin = gui.spin(
            box, self, "random_seed", 0, 2 ** 31 - 1, controlWidth=80,
            label="Fixed seed for random generator:", alignment=Qt.AlignRight,
            callback=self.settings_changed, checked="use_random_seed",
            checkCallback=self.settings_changed)

        # Algorithms
        box = gui.widgetBox(self.controlArea, "Boosting method")
        self.reg_algorithm_combo = gui.comboBox(
            box, self, "loss_index", label="Regression loss function:",
            items=self.losses,
            orientation=Qt.Horizontal, callback=self.settings_changed)

    def create_learner(self):
        if self.base_estimator is None:
            return None
        return self.LEARNER(
            depth=self.depth,
            base_estimator=self.base_estimator,
            iterations=self.n_iterations,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            preprocessors=self.preprocessors)

    @Inputs.learner
    def set_base_learner(self, learner):
        self.Error.no_weight_support.clear()


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWCatBoost()
    ow.resetSettings()
    ow.set_data(Table('iris'))
    ow.show()
    a.exec_()
    ow.saveSettings()