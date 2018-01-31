import random
import unittest

import numpy as np
from Orange.classification import MajorityLearner
from Orange.data import DiscreteVariable, Domain, Table, ContinuousVariable, StringVariable
from Orange.evaluation import CrossValidation
from Orange.widgets.utils.annotated_data import create_annotated_table, ANNOTATED_DATA_FEATURE_NAME

from orangecontrib.d3trax.widgets.precision_recall import cut_off_from_results


class PrecisionRecallTests(unittest.TestCase):

    def setUp(self):
        random.seed(42)

    def test_PR_from_results(self):
        domain = Domain(
            [ContinuousVariable("A"), ContinuousVariable("B")],
            [DiscreteVariable("C", values=["0", "1"])],
            [StringVariable("D")]
        )
        X = np.arange(8).reshape(4, 2)
        Y = np.array([0, 0, 0, 1]).reshape(-1, 1)
        M = np.array(["A", "A", "A", "B"], dtype=object).reshape(-1, 1)
        table = Table.from_numpy(domain, X, Y, M)
        res = CrossValidation(table, [MajorityLearner()], k=2, store_data=True, store_models=True)
        matrix, tpr, ppv = cut_off_from_results(res, 0, .8, 0)

        self.assertEqual(1, tpr)
        self.assertEqual(0.5, ppv)
        np.testing.assert_array_equal([1, 0, 1, 0], matrix)

    def test_create_annotated_table(self):
        domain = Domain(
            [ContinuousVariable("A"), ContinuousVariable("B")],
            [DiscreteVariable("C", values=["0", "1"])],
            [StringVariable("D")]
        )
        X = np.arange(8).reshape(4, 2)
        Y = np.array([1, 0, 0, 1]).reshape(-1, 1)
        M = np.array(["A", "A", "A", "B"], dtype=object).reshape(-1, 1)
        table = Table.from_numpy(domain, X, Y, M)
        mask = Y > 0
        annotated = create_annotated_table(table, mask)

        # check annotated table domain
        self.assertEqual(annotated.domain.variables, domain.variables)
        self.assertEqual(2, len(annotated.domain.metas))
        self.assertIn(domain.metas[0], annotated.domain.metas)
        self.assertIn(ANNOTATED_DATA_FEATURE_NAME,
                      [m.name for m in annotated.domain.metas])

        # check annotated table data
        np.testing.assert_array_equal(annotated.X, table.X)
        np.testing.assert_array_equal(annotated.Y, table.Y)
        np.testing.assert_array_equal(annotated.metas[:, 0].ravel(), table.metas.ravel())
        self.assertEqual(2, np.sum([i[ANNOTATED_DATA_FEATURE_NAME] for i in annotated]))


if __name__ == '__main__':
    unittest.main()
    del PrecisionRecallTests
