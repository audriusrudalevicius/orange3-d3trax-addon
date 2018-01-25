import catboost as cat_ensamble
import tempfile
import inspect
import numpy as np

from Orange.base import SklLearner, Model, Learner
from Orange.classification.base_classification import LearnerClassification, ModelClassification
from Orange.data.filter import HasClass
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import Continuize, RemoveNaNColumns, SklImpute
from Orange.regression.base_regression import ModelRegression, LearnerRegression
from Orange.preprocess.score import LearnerScorer
from Orange.data import Variable, DiscreteVariable

__all__ = ['SklCatBoostClassificationLearner', 'SklCatBoostRegressionLearner', 'CatBoostModel']


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data):
        model = self(data)
        return model._model.feature_importances_


class CatBoostModel(Model):
    used_vals = None

    def __init__(self, _model):
        self._model = _model

    def predict(self, X):
        value = self._model.predict(X)
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X)
            return value, probs
        return value

    def __repr__(self):
        # Params represented as a comment because not passed into constructor
        return super().__repr__() + '  # params=' + repr(self.params)


class CatBoostLearner(Learner, metaclass=WrapperMeta):
    """
    ${skldoc}
    Additional Orange parameters
    preprocessors : list, optional (default=[RemoveNaNClasses(), Continuize(), SklImpute(), RemoveNaNColumns()])
        An ordered list of preprocessors applied to data before
        training or testing.
    """
    __wraps__ = None
    __returns__ = CatBoostModel
    _params = {}
    supports_multiclass = True

    preprocessors = default_preprocessors = [
        HasClass(),
        Continuize(),
        RemoveNaNColumns(),
        SklImpute()]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = self._get_sklparams(value)

    def _get_sklparams(self, values):
        skllearner = self.__wraps__
        if skllearner is not None:
            spec = inspect.getargs(skllearner.__init__.__code__)
            # first argument is 'self'
            assert spec.args[0] == "self"
            params = {name: values[name] for name in spec.args[1:]
                      if name in values}
        else:
            raise TypeError("Wrapper does not define '__wraps__'")
        return params

    def preprocess(self, data):
        data = super().preprocess(data)

        if any(v.is_discrete and len(v.values) > 2
               for v in data.domain.attributes):
            raise ValueError("Wrapped scikit-learn methods do not support " +
                             "multinomial variables.")

        return data

    def __call__(self, data):
        m = super().__call__(data)
        m.params = self.params
        return m

    def fit(self, X, Y, W=None):
        clf = self.__wraps__(**self.params)
        Y = Y.reshape(-1)
        if W is None or not self.supports_weights:
            result = self.__returns__(clf.fit(X, Y))
        else:
            result = self.__returns__(clf.fit(X, Y, sample_weight=W.reshape(-1)))
        return result

    @property
    def supports_weights(self):
        """Indicates whether this learner supports weighted instances.
        """
        return 'sample_weight' in self.__wraps__.fit.__code__.co_varnames

    def __getattr__(self, item):
        try:
            return self.params[item]
        except (KeyError, AttributeError):
            raise AttributeError(item) from None

    # TODO: Disallow (or mirror) __setattr__ for keys in params?

    def __dir__(self):
        dd = super().__dir__()
        return list(sorted(set(dd) | set(self.params.keys())))


# ------------------------------------------------------ #

class CatBoostModelRegression(CatBoostModel, ModelRegression):
    pass


class CatBoostLearnerRegression(CatBoostLearner, LearnerRegression):
    __wraps__ = None
    __returns__ = CatBoostModel
    supports_multiclass = True
    _params = {}

    learner_adequacy_err_msg = "Continuous class variable expected."

    preprocessors = default_preprocessors = [
        HasClass(),
        Continuize(),
        RemoveNaNColumns(),
        SklImpute()]

    def check_learner_adequacy(self, domain):
        return domain.has_continuous_class

    @property
    def params(self):
        return self._params


    @params.setter
    def params(self, value):
        self._params = self._get_sklparams(value)

    def _get_sklparams(self, values):
        skllearner = self.__wraps__
        if skllearner is not None:
            spec = inspect.getargs(skllearner.__init__.__code__)
            # first argument is 'self'
            assert spec.args[0] == "self"
            params = {name: values[name] for name in spec.args[1:]
                      if name in values}
        else:
            raise TypeError("Wrapper does not define '__wraps__'")
        return params

    def preprocess(self, data):
        data = super().preprocess(data)

        if any(v.is_discrete and len(v.values) > 2
               for v in data.domain.attributes):
            raise ValueError("Wrapped scikit-learn methods do not support " +
                             "multinomial variables.")

        return data

    def __call__(self, data):
        m = super().__call__(data)
        m.params = self.params
        return m

    def fit(self, X, Y, W=None):
        clf = self.__wraps__(**self.params)
        Y = Y.reshape(-1)
        if W is None or not self.supports_weights:
            return self.__returns__(clf.fit(X, Y))
        return self.__returns__(clf.fit(X, Y, sample_weight=W.reshape(-1)))

    @property
    def supports_weights(self):
        """Indicates whether this learner supports weighted instances.
        """
        return 'sample_weight' in self.__wraps__.fit.__code__.co_varnames

    def __getattr__(self, item):
        try:
            return self.params[item]
        except (KeyError, AttributeError):
            raise AttributeError(item) from None

    # TODO: Disallow (or mirror) __setattr__ for keys in params?

    def __dir__(self):
        dd = super().__dir__()
        return list(sorted(set(dd) | set(self.params.keys())))


class CatBoostRegressor(CatBoostModelRegression):
    pass


# ------------------------------------------------------ #


class CatBoostModelClassification(CatBoostModel, ModelClassification):
    def __call__(self, data, ret=Model.Value):
        prediction = super().__call__(data, ret=ret)

        if ret == Model.Value:
            return prediction

        if ret == Model.Probs:
            probs = prediction
        else:  # ret == Model.ValueProbs
            value, probs = prediction
            u, i = value.shape
            value = value.reshape((i, u))

        # Expand probability predictions for class values which are not present
        if ret != self.Value:
            n_class = len(self.domain.class_vars)
            max_values = max(len(cv.values) for cv in self.domain.class_vars)
            if max_values != probs.shape[-1]:
                if not self.supports_multiclass:
                    probs = probs[:, np.newaxis, :]
                probs_ext = np.zeros((len(probs), n_class, max_values))
                for c in range(n_class):
                    i = 0
                    class_values = len(self.domain.class_vars[c].values)
                    for cv in range(class_values):
                        if (i < len(self.used_vals[c]) and
                                cv == self.used_vals[c][i]):
                            probs_ext[:, c, cv] = probs[:, c, i]
                            i += 1
                if self.supports_multiclass:
                    probs = probs_ext
                else:
                    probs = probs_ext[:, 0, :]

        if ret == Model.Probs:
            return probs
        else:  # ret == Model.ValueProbs
            return value, probs


class CatBoostLearnerClassification(CatBoostLearner, LearnerClassification):
    __returns__ = CatBoostModelClassification


class CatBoostClassifier(CatBoostModelClassification):
    pass


class SklCatBoostClassificationLearner(CatBoostLearnerClassification, _FeatureScorerMixin):
    __wraps__ = cat_ensamble.CatBoostClassifier
    __returns__ = CatBoostClassifier

    def __init__(self, base_estimator=None, iterations=500, learning_rate=0.03,
                 depth=6, random_seed=None, preprocessors=None, loss_function='MultiClass',
                 calc_feature_importance=True,
                 allow_writing_files=False,
                 train_dir=tempfile.gettempdir()):
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(base_estimator, Fitter):
            base_estimator = base_estimator.get_learner(base_estimator.CLASSIFICATION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SklCatBoostRegressionLearner(CatBoostLearnerRegression, _FeatureScorerMixin):
    __wraps__ = cat_ensamble.CatBoostRegressor
    __returns__ = CatBoostRegressor
    def __init__(self, base_estimator=None, iterations=500, learning_rate=0.03, depth=6,
                 loss_function='RMSE', random_seed=None, preprocessors=None,
                 train_dir=tempfile.gettempdir(), allow_writing_files=False, calc_feature_importance=True):
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(base_estimator, Fitter):
            base_estimator = base_estimator.get_learner(base_estimator.REGRESSION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
