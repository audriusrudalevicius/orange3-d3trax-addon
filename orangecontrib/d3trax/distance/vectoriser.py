import json
import warnings
from abc import abstractmethod, ABC

import numpy as np
from dedupe import serializer, datamodel


def _find_id(field_definition):
    for definition in field_definition:
        if 'identificator' in definition and definition['identificator'] == True:
            return definition['field']
        raise ValueError('Fields mus have one identificator field')


class LabelsVectoriserConfig:
    def __init__(self, field_definition: list, classes: list) -> None:
        self.field_names = list(set([field['field'] for field in field_definition]))
        self.field_definition = field_definition
        self.identificator = _find_id(field_definition)
        self.classes = classes

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, 'r') as f:
            config = json.load(f)[0]
            instance = LabelsVectoriserConfig(
                config['field_definition'],
                config['classes']
            )
        return instance


class Vectoriser(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.data_model = datamodel.DataModel(self.config.field_definition)

    @abstractmethod
    def build_vectors(self, labels) -> list:
        raise NotImplemented


class LabelsVectoriser(Vectoriser):
    @staticmethod
    def read_labels_from_file(input_file):
        with open(input_file, encoding='utf-8') as fp:
            return json.load(fp, cls=serializer.dedupe_decoder)

    def build_vectors(self, labels) -> list:
        self._validate_label_pairs(labels)
        X, Y = self._flatten_labels(labels)

        return self._vectorise(self.data_model, X, Y)

    def _vectorise(self, data_model, examples, y) -> list:
        vectors = []
        distances = data_model.distances(examples)
        i = 0
        for dist_i in range(len(distances)):
            vector = {}
            dist = distances[dist_i]
            item_class = y[dist_i]
            pair = examples[dist_i]
            for f, d in enumerate(dist):
                vector[data_model._variables[f].name] = d
            vector['index'] = i
            vector['klass'] = item_class
            vector['id1'] = pair[0][self.config.identificator]
            vector['id2'] = pair[1][self.config.identificator]
            vectors.append(vector)
            i += 1

        return vectors

    def _flatten_labels(self, training_pairs):
        examples = []
        y = []
        for label, pairs in training_pairs.items():
            for pair in pairs:
                y.append(self.config.classes.index(label))
                examples.append(pair)

        return examples, np.array(y)

    def _validate_label_pairs(self, labeled_pairs):
        try:
            labeled_pairs.items()
            for klass in self.config.classes:
                # noinspection PyStatementEffect
                labeled_pairs[klass]
        except:
            raise ValueError(
                'labeled_pairs must be a dictionary with keys {}'.format(', '.join(self.config.classes))
            )
        validated_classes = 0
        for klass in self.config.classes:
            if labeled_pairs[klass]:
                pair = labeled_pairs[klass][0]
                self._validate_record_pair(pair)
                validated_classes += 1

        if validated_classes == 0:
            warnings.warn("Didn't return any labeled record pairs")

    def _validate_record_pair(self, record_pair):
        try:
            record_pair[0]
        except:
            raise ValueError("The elements of data_sample must be pairs "
                             "of record_pairs (ordered sequences of length 2)")

        if len(record_pair) != 2:
            raise ValueError("The elements of data_sample must be pairs of record_pairs")
        try:
            record_pair[0].keys() and record_pair[1].keys()
        except:
            raise ValueError("A pair of record_pairs must be made up of two dictionaries ")

        self.data_model.check(record_pair[0])
        self.data_model.check(record_pair[1])

