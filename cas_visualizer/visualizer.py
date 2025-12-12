import abc

import pandas as pd

from cas_visualizer.util import load_typesystem
from cassis import Cas, TypeSystem
from cassis.typesystem import FeatureStructure
from spacy.displacy import EntityRenderer, DependencyRenderer, SpanRenderer
from typing import Any, Dict, Literal


class Visualizer(abc.ABC):
    def __init__(self, ts: TypeSystem):
        self._cas = None
        self._ts = None
        self._types = set()
        self._colors = dict()
        self._labels = dict()
        self._features = dict()
        self._feature_colors = dict() # (name, feature) -> color
        self._feature_labels = dict()
        self._value_labels = dict() #(name, feature, value) -> label
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

    @property
    def values_to_labels(self) -> dict:
        return self._value_labels

    @abc.abstractmethod
    def render_visualization(self):
        """Generates the visualization based on the provided configuration."""
        raise NotImplementedError

    def add_type(self,
                 name: str,
                 feature: str = None,
                 color: str = None,
                 label: str = None,
                 ):
        """
        Adds a new annotation type to the visualizer.
        :param name: name of the annotation type as declared in the type system.
        :param feature: optionally, the value of a feature can be used as the tag label of the visualized annotation
        :param color: optionally, a specific string color name for the annotation
        :param label: optionally, a specific string label for the annotation (defaults to type_name)
        """
        if not name:
            raise TypeError('type path cannot be empty')
        self._types.add(name)
        self._colors[name] = color if color else next(self._default_colors)
        self._labels[name] = label if label else name.split('.')[-1]
        if feature:
            self._add_feature_by_type(name, feature)

    def add_feature(self,
                 name: str,
                 feature:str,
                 value,
                 color: str = None,
                 label: str = None,
                 ):
        """
        Adds a new annotation type to the visualizer.
        :param name: name of the annotation type as declared in the type system.
        :param feature: the feature of the annotation type
        :param value: the value of the feature to annotate
        :param color: optionally, the color for the annotation of a specific feature value
        :param label: optionally, replaces the specified value in the annotation labels
        """
        if not name:
            raise VisualizerException('type name cannot be empty')
        self._types.add(name)
        if feature:
            self._add_feature_by_type(name, feature)
            if value:
                if label:
                    self._value_labels[(name, feature, value)] = label
                    self._feature_colors[(name, label)] = color if color else next(self._default_colors)
                else:
                    self._feature_colors[(name, value)] = color if color else next(self._default_colors)
            else:
                raise VisualizerException(f'a value for feature {feature} must be specified')
        else:
            raise VisualizerException(f'a feature for type {name} must be specified')

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
            label = item.get('label')
            self.add_type(type_path, feature_name, color, label)

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
            keys = [key for key in self._value_labels.keys() if key[0] == type_path]
            for key in keys:
                self._value_labels.pop(key)

        except:
            raise VisualizerException('type path cannot be found')

    def visualize(self, cas: Cas, view_name=None):
        self._cas = cas
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
        default_label = self.values_to_labels.get((annotation_type,annotation_feature,feature_value))
        if default_label:
            return default_label
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

class SpacyDependencyVisualizer(Visualizer):

    T_DEPENDENCY = 'org.dakoda.syntax.UDependency'
    T_POS = 'de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS'
    T_SENTENCE = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'

    output_formats = Literal['html', 'svg', 'pdf', 'png']

    def __init__(self, ts: TypeSystem,
                 dep_type: str = T_DEPENDENCY,
                 pos_type: str = T_POS,
                 span_type: str = T_SENTENCE,
                 ):
        """

        :param ts: TypeSystem to use.
        :param dep_type: Type used to determine the dependencies.
        :param pos_type: Type used to determine the part-of-speech.
        :param span_type: Type used to determine the spans.
        """
        super().__init__(ts)
        self._dep_type = dep_type
        self._pos_type = pos_type
        self._span_type = span_type

    def visualize(self, cas: Cas,
                  minify: bool = False,
                  options: Dict[str, Any] = {},
                  page: bool = False,
                  start: int = 0,
                  end: int = -1,
                  output_format:output_formats = 'html',
                  ):
        """

        :param cas: CAS object to visualize.
        :param minify: optionally, minifies HTML markup.
        :param options: optionally, specifies parameters for spacy rendering. Supported options are: fine_grained,
            add_lemma, collapse_punct, collapse_phrases, compact, color, bg, font, offset_x, arrow_stroke, arrow_width,
            arrow_spacing, word_spacing, distance
        :param page: optionally, render parses wrapped as full HTML page.
        :param start: optionally, specifies starting position of spans.
        :param end: optionally, specifies ending position of spans.
        :param view_name: optionally, specifies name of the view being rendered.
        :param output_format: optionally, specifies output format. Supported options: html, pdf, svg, html.
        :return: dependency graph as specified by output_format.
        """
        self._minify = minify
        self._options = options
        self._output_format = output_format
        self._page = page
        self._span_range = [start, end]
        if end > -1 and start > end:
            raise VisualizerException(f'Given span range [start={start}, end={end}] is not valid.')
        return super().visualize(cas)


    def render_visualization(self):
        parsed = []
        renderer = DependencyRenderer(options=self._options)
        span_start = self._span_range[0]
        span_end = self._span_range[1]
        if span_end == -1:
            span_end = len(self._cas.sofa_string)
        for item in self._cas.select(self._span_type):
            if self._span_range is None or (item.begin >= span_start and item.end <= span_end):
                struct = self.dep_to_dict(covered=item)
                parsed.append({"words": struct['words'], "arcs": struct['arcs']})

        if len(parsed) == 0:
            raise VisualizerException(f'No spans found for type {self._span_type} in range {self._span_range}.')

        match self._output_format:
            case 'html':
                return renderer.render(parsed, page=self._page, minify=self._minify)
            case 'svg':
                rendered = []
                for i, p in enumerate(parsed):
                    svg = renderer.render_svg(f"render_id-{i}", p["words"], p["arcs"])
                    rendered.append(svg)
                return rendered
            case _:
                raise VisualizerException(f'Output format {self._output_format} is not yet supported.')


    def dep_to_dict(self, covered: FeatureStructure):

        # construct dummy annotation to use for covered select (i.e. restrict results to given span (usually a single sentence))
        #T = cas.typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token')
        #covered = T(begin=span[0], end=span[1])

        offset_to_index = {}
        for idx, p in enumerate(self._cas.select(self._pos_type)):
            offset_to_index[p.begin] = idx

        words = [
            {
                'text': p.get_covered_text(),
                'tag': p.PosValue
            }
            for p in self._cas.select_covered(
                self._pos_type,
                covering_annotation=covered
            )
        ]

        # how to restrict arcs to covered span?
        arcs = [
            {
                'start': offset_to_index[d.Governor.begin],
                'end': offset_to_index[d.Dependent.begin],
                'label': d.DependencyType,
                'dir': 'right' if d.Governor.begin < d.Dependent.begin else 'left',
            }
            for d in self._cas.select(
                self._dep_type,
            )
            #if span[0] <= d.Governor.begin <= span[1] and span[0] <= d.Dependent.begin <= span[1]
        ]

        # ensure that start is always smaller than end
        # i.e. direction is only encoded in 'dir' field
        for arc in arcs:
            if arc['start'] > arc['end']:
                arc['start'], arc['end'] = arc['end'], arc['start']

        # remove root (i.e. keep everything except root where start == end)
        arcs = [arc for arc in arcs if arc['start'] != arc['end']]

        return {"words": words, "arcs": arcs}

