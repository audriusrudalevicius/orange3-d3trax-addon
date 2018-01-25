from orangecontrib.d3trax.ensembles import (
    SklCatBoostClassificationLearner, SklCatBoostRegressionLearner
)
from Orange.modelling import Model, Fitter
from Orange.preprocess.score import LearnerScorer
from Orange.data import Variable, DiscreteVariable

__all__ = ['CatBoostLearner']


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data):
        model = self(data)
        return model._model.feature_importances_


class CatBoostLearner(Fitter, _FeatureScorerMixin):
    __fits__ = {'classification': SklCatBoostClassificationLearner,
                'regression': SklCatBoostRegressionLearner}

    __returns__ = Model
