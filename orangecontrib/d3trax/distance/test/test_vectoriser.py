import unittest

from orangecontrib.d3trax.distance.vectoriser import LabelsVectoriserConfig, LabelsVectoriser


class VectoriserTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def test_labels_from_file(self):
        config = LabelsVectoriserConfig.load_config('test_config.json')
        instance = LabelsVectoriser(config)
        labels = LabelsVectoriser.read_labels_from_file('test_sample.json')
        vectors = instance.build_vectors(labels)

        self.assertEqual(2, len(vectors))
        self.assertEqual(8, len(vectors[0]))
        self.assertEqual(8, len(vectors[1]))

        self.assertTrue('index' in vectors[0])
        self.assertTrue('klass' in vectors[0])
        self.assertTrue('id1' in vectors[0])
        self.assertTrue('id2' in vectors[0])

        self.assertTrue('index' in vectors[1])
        self.assertTrue('klass' in vectors[1])
        self.assertTrue('id1' in vectors[1])
        self.assertTrue('id2' in vectors[1])
        pass


if __name__ == '__main__':
    unittest.main()
    del VectoriserTests
