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
        self._feature_colors = dict()
        self._feature_labels = dict()
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
    def features_to_colors(self) -> dict:
        return self._feature_colors

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

    def add_type(self,
                 name: str,
                 feature: str = None,
                 color: str = None,
                 default_label: str = None,
                 ):
        """
        Adds a new annotation type to the visualizer.
        :param name: name of the annotation type as declared in the type system.
        :param feature: optionally, the value of a feature can be used as the tag label of the visualized annotation
        :param color: optionally, a specific string color name for the annotation
        :param default_label: optionally, a specific string label for the annotation (defaults to type_name)
        """
        if not name:
            raise TypeError('type path cannot be empty')
        self._types.add(name)
        self._colors[name] = color if color else next(self._default_colors)
        self._labels[name] = default_label if default_label else name.split('.')[-1]
        if feature:
            self._add_feature_by_type(name, feature)

    def add_feature(self,
                 name: str,
                 feature:str = None,
                 value = None,
                 color: str = None,
                 ):
        """
        Adds a new annotation type to the visualizer.
        :param name: name of the annotation type as declared in the type system.
        :param feature: optionally, the value of a feature can be used as the tag label of the visualized annotation
        :param value: optionally, the value of a feature can determine a different color for the annotation
        :param color: optionally, the color for the annotation of a specific feature value
        """
        if not name:
            raise TypeError('type path cannot be empty')
        self._types.add(name)
        if feature:
            self._add_feature_by_type(name, feature)
            if value is not None:
                self._feature_colors[(name, value)] = color if color else next(self._default_colors)

    def _add_feature_by_type(self, type_name, feature_name):
        current_feature = self._features.get(type_name)
        if current_feature is not None and current_feature != feature_name:
            # new feature replaces current feature -> remove selected color
            remove_list = []
            for key in self._feature_colors.keys():
                if key[0] == type_name:
                    remove_list.append(key)
            for key in remove_list:
                del self._feature_colors[key]
        self._features[type_name] = feature_name

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
            keys = [key for key in self._feature_colors.keys() if key[0] == type_path]
            for key in keys:
                self._feature_colors.pop(key)

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
                feature_value = Visualizer.get_feature_value(fs, self.types_to_features.get(type_item))
                records.append({
                    'text': fs.get_covered_text(),
                    'feature': self.types_to_features.get(type_item),
                    'value': feature_value,
                    'begin': fs.begin,
                    'end': fs.end,
                })

        return pd.DataFrame.from_records(records).sort_values(by=['begin', 'end'])


class SpanVisualizer(Visualizer):
    HIGHLIGHT = 'HIGHLIGHT'
    UNDERLINE = 'UNDERLINE'

    def __init__(self, ts: TypeSystem, span_type: str=None, types: list[str]=None):
        super().__init__(ts)
        self._span_types = [SpanVisualizer.HIGHLIGHT, SpanVisualizer.UNDERLINE]
        self._selected_span_type = SpanVisualizer.UNDERLINE
        if span_type is not None:
            self.selected_span_type = span_type
        self._allow_highlight_overlap = False
        if types is not None:
            for type_name in types:
                self.add_type(type_name)

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

    def get_label(self, fs: FeatureStructure, annotation_type):
        annotation_feature = self.types_to_features.get(annotation_type)
        feature_value = Visualizer.get_feature_value(fs, annotation_feature)
        return feature_value if feature_value is not None else self.types_to_labels.get(annotation_type)

    def get_color(self, annotation_type, label):
        label_color = self.features_to_colors.get((annotation_type, label))
        return label_color if label_color else self.types_to_colors.get(annotation_type)

    def parse_ents(self):  # see parse_ents spaCy/spacy/displacy/__init__.py
        tmp_ents = []
        labels_to_colors = dict()
        for annotation_type in self.type_list:
            for fs in self._cas.select(annotation_type):
                label = self.get_label(fs, annotation_type)
                color = self.get_color(annotation_type, label)
                if color:
                    # a color is required for each annotation
                    tmp_ents.append(
                        {
                            "start": fs.begin,
                            "end": fs.end,
                            "label": label,
                        }
                    )
                    labels_to_colors[label] = color
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

    def create_spans(self,
                     cas_sofa_tokens: list,
                     annotation_type: str,
                     ) -> list[dict[str, str]]:
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
                    "label": self.get_label(fs, annotation_type),
                }
            )
        return tmp_spans

    def parse_spans(self) -> str:  # see parse_ents spaCy/spacy/displacy/__init__.py
        selected_annotations = [item for typeclass in self.type_list for item in self._cas.select(typeclass)]
        tmp_tokens = self.create_tokens(self._cas.sofa_string, selected_annotations)
        tmp_token_texts = [_["text"] for _ in sorted(tmp_tokens, key=lambda t: t["start"])]

        tmp_spans = []
        labels_to_colors = dict()
        for annotation_type in self.type_list:
            for tmp_span in self.create_spans(tmp_tokens, annotation_type):
                label = tmp_span["label"]
                color = self.get_color(annotation_type, label)
                if color is not None:
                    # remove spans without a color from list
                    labels_to_colors[label] = color
                    tmp_spans.append(tmp_span)
        tmp_spans.sort(key=lambda x: x["start"])
        return SpanRenderer({"colors": labels_to_colors}).render_spans(tmp_token_texts, tmp_spans, "")
