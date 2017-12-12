from orangecontrib.d3trax.ensembles import (
    SklCatBoostClassificationLearner, SklCatBoostRegressionLearner
)
from Orange.modelling import Model, Fitter

__all__ = ['CatBoostLearner']


class CatBoostLearner(Fitter):
    __fits__ = {'classification': SklCatBoostClassificationLearner,
                'regression': SklCatBoostRegressionLearner}

    __returns__ = Model
