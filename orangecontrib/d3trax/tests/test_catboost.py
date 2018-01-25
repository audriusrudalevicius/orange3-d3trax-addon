import unittest
import Orange

from orangecontrib.d3trax.ensembles import SklCatBoostRegressionLearner, SklCatBoostClassificationLearner


class CatboostTests(unittest.TestCase):
    def test_regression_voting(self):
        data = Orange.data.Table("servo")
        lr = SklCatBoostRegressionLearner()
        res = Orange.evaluation.CrossValidation(data, [lr], k=5)

        self.assertEquals(res.failed, [False])

    def test_classification_wine(self):
        data = Orange.data.Table("iris")
        lr = SklCatBoostClassificationLearner()
        res = Orange.evaluation.CrossValidation(data, [lr], k=5)

        self.assertEquals(res.failed, [False])

    def test_score(self):
        data = Orange.data.Table("iris")
        lr = SklCatBoostClassificationLearner()
        scores = lr.score(data)

        self.assertEquals(len(scores), 4)


if __name__ == "__main__":
    unittest.main()