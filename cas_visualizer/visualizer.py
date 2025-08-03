import abc
import pandas as pd

from cas_visualizer.util import cas_from_string, load_typesystem
from cassis import Cas, TypeSystem
from cassis.typesystem import FeatureStructure
from spacy.displacy import EntityRenderer, SpanRenderer


class Visualizer(abc.ABC):
    def __init__(self, ts: TypeSystem):
        self._cas = None
        self._ts = None
        self._types = set()
        self._colors = dict()
        self._labels = dict()
        self._features = dict()
        self._default_colors = iter(["lightgreen", "orangered", "orange", "plum", "palegreen", "mediumseagreen",
                       "steelblue", "skyblue", "navajowhite", "mediumpurple", "rosybrown", "silver", "gray",
                       "paleturquoise"])
        match ts:
            case str():
                self._ts = load_typesystem(ts)
            case TypeSystem():
                self._ts = ts
            case _:
                raise VisualizerException('typesystem cannot be None')

    @property
    def types_to_colors(self) -> dict:
        return self._colors

    @property
    def types_to_features(self) -> dict:
        return self._features

    @property
    def types_to_labels(self) -> dict:
        return self._labels

    @property
    def type_list(self) -> list:
        return list(self._types)

    @abc.abstractmethod
    def render_visualization(self):
        """Generates the visualization based on the provided configuration."""
        raise NotImplementedError

    def add_type(self, type_name, feature_name=None, color=None, default_label=None):
        """
        Adds a new annotation type to the visualizer.
        :param type_name: name of the annotation type as declared in the type system.
        :param feature_name: optionally, the value of a feature can be used as the tag label of the visualized annotation
        :param color: optionally, a specific string color name for the annotation
        :param default_label: optionally, a specific string label for the annotation (defaults to type_name)
        """
        if type_name is None or len(type_name) == 0:
            raise TypeError('type path cannot be empty')
        self._types.add(type_name)
        self._features[type_name] = feature_name
        if color is None:
            color = next(self._default_colors)
        self._colors[type_name] = color
        if default_label is None or len(default_label) == 0:
            default_label = type_name.split('.')[-1]
        self._labels[type_name] = default_label

    def add_types_from_list_of_dict(self, config_list: list):
        for item in config_list:
            type_path = item.get('type_path')
            feature_name = item.get('feature_name')
            color = item.get('color')
            default_label = item.get('default_label')
            self.add_type(type_path, feature_name, color, default_label)

    @staticmethod
    def get_feature_value(fs:FeatureStructure, feature_name:str):
        return fs.get(feature_name) if feature_name is not None else None

    def remove_type(self, type_path):
        if type_path is None:
            raise VisualizerException('type path cannot be empty')
        try:
            self._types.remove(type_path)
            self._colors.pop(type_path)
            self._labels.pop(type_path)
            self._features.pop(type_path)
        except:
            raise VisualizerException('type path cannot be found')

    def visualize(self, cas: Cas|str):
        match cas:
            case str():
                self._cas = cas_from_string(cas, self._ts)
            case Cas():
                self._cas = cas
            case _:
                raise VisualizerException('cas cannot be None')
        return self.render_visualization()

class VisualizerException(Exception):
    pass


class TableVisualizer(Visualizer):
    def render_visualization(self):
        records = []
        for type_item in self.type_list:
            for fs in self._cas.select(type_item):
                feature_value = Visualizer.get_feature_value(fs, self.types_to_features[type_item])
                records.append({
                    'text': fs.get_covered_text(),
                    'feature': self.types_to_features[type_item],
                    'value': feature_value,
                    'begin': fs.begin,
                    'end': fs.end,
                })

        return pd.DataFrame.from_records(records).sort_values(by=['begin', 'end'])


class SpanVisualizer(Visualizer):
    HIGHLIGHT = 'HIGHLIGHT'
    UNDERLINE = 'UNDERLINE'

    def __init__(self, ts: TypeSystem):
        super().__init__(ts)
        self._span_types = [SpanVisualizer.HIGHLIGHT, SpanVisualizer.UNDERLINE]
        self._selected_span_type = SpanVisualizer.UNDERLINE
        self._allow_highlight_overlap = False

    @property
    def selected_span_type(self):
        return self._selected_span_type

    @selected_span_type.setter
    def selected_span_type(self, value:str):
        if value not in self._span_types:
            raise VisualizerException('Invalid span type', value, 'Expected one of', self._span_types)
        self._selected_span_type = value

    @property
    def allow_highlight_overlap(self):
        return self._allow_highlight_overlap

    @allow_highlight_overlap.setter
    def allow_highlight_overlap(self, value:bool):
        self._allow_highlight_overlap = value

    def render_visualization(self):
        match self.selected_span_type:
            case SpanVisualizer.HIGHLIGHT:
                return self.parse_ents()
            case SpanVisualizer.UNDERLINE:
                return self.parse_spans()
            case _:
                raise VisualizerException('Invalid span type')

    @staticmethod
    def get_label(fs: FeatureStructure, annotation_label, annotation_feature):
        #if annotation_feature is not None and fs.get(annotation_feature) is not None and len(
        #        fs[annotation_feature]) > 0:
        #    return fs[annotation_feature]
        feature_value = Visualizer.get_feature_value(fs, annotation_feature)
        return feature_value if feature_value is not None else annotation_label

    def parse_ents(self):  # see parse_ents spaCy/spacy/displacy/__init__.py
        tmp_ents = []
        labels_to_colors = dict()
        for annotation_type, annotation_label in self.types_to_labels.items():
            annotation_feature = self.types_to_features[annotation_type]
            for fs in self._cas.select(annotation_type):
                label = self.get_label(fs, annotation_label, annotation_feature)
                tmp_ents.append(
                    {
                        "start": fs.begin,
                        "end": fs.end,
                        "label": label,
                    }
                )
                labels_to_colors[label] = self.types_to_colors[annotation_type]
        tmp_ents.sort(key=lambda x: (x['start'], x['end']))
        if not self._allow_highlight_overlap and self.check_overlap(tmp_ents):
            raise VisualizerException(
                'The highlighted annotations are overlapping. Choose a different set of annotations or set the allow_highlight_overlap parameter to True.')

        return EntityRenderer({"colors": labels_to_colors}).render_ents(self._cas.sofa_string, tmp_ents, "")

    # requires a sorted list of "tmp_ents" as returned by tmp_ents.sort(key=lambda x: (x['start'], x['end']))
    @staticmethod
    def check_overlap(l_ents):
        for i in range(len(l_ents)):
            start_i = l_ents[i]['start']
            for j in range(len(l_ents)):
                if i != j:
                    start_j = l_ents[j]['start']
                    end_j = l_ents[j]['end']
                    if start_j <= start_i < end_j:
                        return True
        return False

    @staticmethod
    def create_tokens(cas_sofa_string: str, feature_structures: list[FeatureStructure]) -> list[dict[str, str]]:
        cas_sofa_tokens = []
        cutting_points = set(_['begin'] for _ in feature_structures).union(_['end'] for _ in feature_structures)
        char_index_after_whitespace = set([i + 1 for i, char in enumerate(cas_sofa_string) if char.isspace()])
        cutting_points = cutting_points.union(char_index_after_whitespace)
        prev_point = point = 0
        for point in sorted(cutting_points):
            if point != 0:
                tmp_token = {"start": prev_point, "end": point, "text": cas_sofa_string[prev_point:point]}
                cas_sofa_tokens.append(tmp_token)
                prev_point = point
        if point < len(cas_sofa_string):
            tmp_token = {"start": prev_point, "end": len(cas_sofa_string), "text": cas_sofa_string[prev_point:]}
            cas_sofa_tokens.append(tmp_token)
        return cas_sofa_tokens

    def create_spans(self, cas_sofa_tokens: list, annotation_type: str, annotation_feature: str,
                     annotation_label: str) -> list[dict[str, str]]:
        tmp_spans = []
        for fs in self._cas.select(annotation_type):
            start_token = 0
            end_token = len(cas_sofa_tokens)
            for idx, token in enumerate(cas_sofa_tokens):
                if token["start"] == fs.begin:
                    start_token = idx
                if token["end"] == fs.end:
                    end_token = idx + 1

            tmp_spans.append(
                {
                    "start": fs.begin,
                    "end": fs.end,
                    "start_token": start_token,
                    "end_token": end_token,
                    "label": self.get_label(fs, annotation_label, annotation_feature),
                }
            )
        return tmp_spans

    def parse_spans(self) -> str:  # see parse_ents spaCy/spacy/displacy/__init__.py
        selected_annotations = [item for typeclass in self.type_list for item in self._cas.select(typeclass)]
        tmp_tokens = self.create_tokens(self._cas.sofa_string, selected_annotations)
        tmp_token_texts = [_["text"] for _ in sorted(tmp_tokens, key=lambda t: t["start"])]

        tmp_spans = []
        labels_to_colors = dict()
        for annotation_type, annotation_label in self.types_to_labels.items():
            annotation_feature = self.types_to_features[annotation_type]
            new_spans = self.create_spans(tmp_tokens, annotation_type, annotation_feature, annotation_label)
            for span in new_spans:
                label = span["label"]
                labels_to_colors[label] = self.types_to_colors[annotation_type]
            tmp_spans.extend(new_spans)
        tmp_spans.sort(key=lambda x: x["start"])
        return SpanRenderer({"colors": labels_to_colors}).render_spans(tmp_token_texts, tmp_spans, "")



