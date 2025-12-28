import abc
import pandas as pd
from cas_visualizer.util import load_typesystem
from cassis import Cas, TypeSystem
from cassis.typesystem import FeatureStructure
from collections.abc import Iterator
from itertools import cycle
from spacy.displacy import EntityRenderer, DependencyRenderer, SpanRenderer
from typing import Any, Dict, Optional, Literal, Union

OutputFormat = Literal['html', 'svg', 'pdf', 'png']

class VisualizerException(Exception):
    pass

class Visualizer(abc.ABC):
    def __init__(self, ts: str|TypeSystem):
        self._types: set[str] = set()
        self._colors: dict[str, str] = dict()   # type name to color
        self._labels: dict[str, str] = dict()
        self._features: dict[str, str] = dict()
        self._feature_colors: dict[tuple[str, str], str] = dict() # (name, feature) -> color
        self._feature_labels: dict[tuple[str, str], str] = dict()
        self._value_labels: dict[tuple[str, Optional[str], Any], str] = dict() #(name, feature, value) -> label
        
        # Default color iterator
        color_list = [
            "lightgreen", "orangered", "orange", "plum", "palegreen",
            "mediumseagreen", "steelblue", "skyblue", "navajowhite",
            "mediumpurple", "rosybrown", "silver", "gray", "paleturquoise"
        ]
        self._default_colors: Iterator[str] = cycle(color_list)

        if isinstance(ts, str):
            self._ts = load_typesystem(ts)
        elif isinstance(ts, (TypeSystem)):
            self._ts = ts
        else:
            raise VisualizerException('typesystem must be a string path or TypeSystem object')

    @property
    def features_to_colors(self) -> dict[tuple[str,str],str]:
        return self._feature_colors

    @property
    def types_to_colors(self) -> dict[str, str]:
        return self._colors

    @property
    def types_to_features(self) -> dict[str, str]:
        return self._features

    @property
    def types_to_labels(self) -> dict[str, str]:
        return self._labels

    # TODO why list and not set?
    @property
    def type_list(self) -> list[str]:
        return list(self._types)

    @property
    def values_to_labels(self) -> dict[tuple[str, Optional[str], Any], str]:
        return self._value_labels

    def add_type(self,
                 name: str,
                 feature: str | None = None,
                 color: str | None  = None,
                 label: str | None  = None,
                 ):
        """
        Adds a new annotation type to the visualizer.
        :param name: name of the annotation type as declared in the type system.
        :param feature: optionally, the value of a feature can be used as the tag label of the visualized annotation
        :param color: optionally, a specific string color name for the annotation
        :param label: optionally, a specific string label for the annotation (defaults to type_name)
        """
        if not name:
            raise VisualizerException('type path cannot be empty')
        self._types.add(name)
        self._colors[name] = color if color else next(self._default_colors)
        self._labels[name] = label if label else name.split('.')[-1]
        if feature:
            self._add_feature_by_type(name, feature)

    def add_feature(
        self,
        name: str,
        feature: str,
        value: Any,
        color: str | None = None,
        label: str | None = None,
    ):
        if not name:
            raise VisualizerException('type name cannot be empty')
        self._types.add(name)
        if not feature:
            raise VisualizerException(f'a feature for type {name} must be specified')

        self._add_feature_by_type(name, feature)

        if value is None:
            raise VisualizerException(f'a value for feature {feature} must be specified')

        eff_label = label if label is not None else str(value)
        self._value_labels[(name, feature, value)] = eff_label
        self._feature_colors[(name, eff_label)] = color if color else next(self._default_colors)

    @abc.abstractmethod
    def visualize(self, cas: Cas) -> Any:
        """Generates the visualization based on the provided configuration."""
        raise NotImplementedError

    def _add_feature_by_type(self, type_name: str, feature_name: str):
        current_feature = self._features.get(type_name)
        if current_feature is not None and current_feature != feature_name:
            # new feature replaces current feature -> remove selected color
            remove_list: list[tuple[str, str]] = []
            for key in self._feature_colors.keys():
                if key[0] == type_name:
                    remove_list.append(key)
            for key in remove_list:
                del self._feature_colors[key]
        self._features[type_name] = feature_name

    # # TODO is this ever used?
    # def add_types_from_list_of_dict(self, config_list: list):
    #     for item in config_list:
    #         type_path = item.get('type_path')
    #         feature_name = item.get('feature_name')
    #         color = item.get('color')
    #         label = item.get('label')
    #         self.add_type(type_path, feature_name, color, label)

    def _get_feature_value(self, fs: FeatureStructure, feature_name: Optional[str]) -> Any:
        if feature_name is None:
            return None
        try:
            return fs.get(feature_name)
        except KeyError:
            # Optional, aber hilfreicher als still None zurückzugeben
            raise VisualizerException(
                f"Feature '{feature_name}' not found on type '{fs.type.name}'"
            )

    # TODO do we need that, why would we want to remove types?
    # implement clear() or similar instead?
    def remove_type(self, type_path: str):
        if not type_path:
            raise VisualizerException('type path cannot be empty')
        self._types.discard(type_path)
        self._colors.pop(type_path, None)
        self._labels.pop(type_path, None)
        self._features.pop(type_path, None)
        keys = [key for key in self._feature_colors.keys() if key[0] == type_path]
        for key in keys:
            self._feature_colors.pop(key)
        keys = [key for key in self._value_labels.keys() if key[0] == type_path]
        for key in keys:
            self._value_labels.pop(key)

TableFormat = Literal['html', 'csv', 'json', 'latex']

class TableVisualizer(Visualizer):
    def __init__(
        self,
        ts: str | TypeSystem,
        *,
        default_format: TableFormat = 'html',
        default_render_options: Optional[Dict[str, Any]] = None,
        default_sort: bool = True,
    ):
        super().__init__(ts)
        self._default_format = default_format
        self._default_render_options = default_render_options or {}
        self._default_sort = default_sort

    def build(self, cas: Cas, begin: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:
        if end is None:
            end = len(cas.sofa_string)
        cols = ['text', 'feature', 'value', 'begin', 'end']
        records: list[dict[str, Any]] = []
        for type_item in self.type_list:
            for fs in cas.select(type_item):
                if begin is not None and fs.begin < begin:
                    continue
                if end is not None and fs.end > end:
                    continue
                feature_name = self.types_to_features.get(type_item)
                feature_value = self._get_feature_value(fs=fs, feature_name=feature_name)
                records.append({
                    'text': fs.get_covered_text(),
                    'feature': feature_name,
                    'value': feature_value,
                    'begin': fs.begin,
                    'end': fs.end,
                })
        df = pd.DataFrame.from_records(records, columns=cols)
        if self._default_sort and not df.empty:
            df = df.sort_values(by=['begin', 'end'], kind='mergesort')
        return df

    def render(
        self,
        spec: pd.DataFrame,
        *,
        format: Optional[TableFormat] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        fmt = format or self._default_format
        opts = {**self._default_render_options, **(render_options or {})}
        if fmt == 'html':
            return spec.to_html(**({'index': False, 'escape': True} | opts))
        if fmt == 'csv':
            return spec.to_csv(**({'index': False} | opts))
        if fmt == 'json':
            return spec.to_json(**({'orient': 'records'} | opts))
        if fmt == 'latex':
            return spec.to_latex(**({'index': False, 'escape': True} | opts))
        raise VisualizerException(f"Unsupported table output format: {fmt}")

    def visualize(
        self,
        cas: Cas,
        begin: Optional[int] = None,
        end: Optional[int] = None,
        *,
        format: Optional[TableFormat] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        spec = self.build(cas, begin=begin, end=end)
        return self.render(spec, format=format, render_options=render_options)

class SpanVisualizer(Visualizer):
    HIGHLIGHT: str = 'HIGHLIGHT'
    UNDERLINE: str = 'UNDERLINE'

    def __init__(self, ts: str|TypeSystem, span_type: Optional[str]=None):
        super().__init__(ts)
        self._span_types: list[str] = [SpanVisualizer.HIGHLIGHT, SpanVisualizer.UNDERLINE]
        self._selected_span_type = SpanVisualizer.UNDERLINE
        if span_type is not None:
            self.selected_span_type = span_type
        self._allow_highlight_overlap = False

    @property
    def selected_span_type(self) -> str:
        return self._selected_span_type

    @selected_span_type.setter
    def selected_span_type(self, value: str) -> None:
        if value not in self._span_types:
            raise VisualizerException(f'Invalid span type: {value}. Expected one of {self._span_types}')
        self._selected_span_type = value

    @property
    def allow_highlight_overlap(self) -> bool:
        return self._allow_highlight_overlap

    @allow_highlight_overlap.setter
    def allow_highlight_overlap(self, value:bool):
        self._allow_highlight_overlap = value

    def visualize(self, cas: Cas):
        match self.selected_span_type:
            case SpanVisualizer.HIGHLIGHT:
                return self._parse_ents(cas)
            case SpanVisualizer.UNDERLINE:
                return self._parse_spans(cas)
            case _:
                raise VisualizerException('Invalid span type')

    def _get_label(self, fs: FeatureStructure, annotation_type: str) -> Optional[str]:
        annotation_feature = self.types_to_features.get(annotation_type)
        feature_value = self._get_feature_value(fs=fs, feature_name=annotation_feature)
        default_label = self.values_to_labels.get((annotation_type, annotation_feature, feature_value))
        if default_label:
            return default_label
        return str(feature_value) if feature_value is not None else self.types_to_labels.get(annotation_type)

    def _get_color(self, annotation_type, label):
        label_color = self.features_to_colors.get((annotation_type, label))
        return label_color if label_color else self.types_to_colors.get(annotation_type)

    def _parse_ents(self, cas: Cas):  # see parse_ents spaCy/spacy/displacy/__init__.py
        tmp_ents = []
        labels_to_colors = dict()
        for annotation_type in self.type_list:
            for fs in cas.select(annotation_type):
                label = self._get_label(fs, annotation_type)
                color = self._get_color(annotation_type, label)
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
        
        sorted_ents = sorted(tmp_ents, key=lambda x: (x['start'], x['end']))

        if not self._allow_highlight_overlap and self._check_overlap(sorted_ents):
            raise VisualizerException(
                'The highlighted annotations are overlapping. Choose a different set of annotations or set the allow_highlight_overlap parameter to True.')

        return EntityRenderer({"colors": labels_to_colors}).render_ents(cas.sofa_string, sorted_ents, "")

    # requires a sorted list of "tmp_ents" as returned by tmp_ents.sort(key=lambda x: (x['start'], x['end']))
    @staticmethod
    def _check_overlap(sorted_ents: list[dict]) -> bool:
        max_end = -1
        for ent in sorted_ents:
            s = ent['start']
            e = ent['end']
            if s < max_end:
                return True
            if e > max_end:
                max_end = e
        return False

    @staticmethod
    def _split_text_for_spans(
        cas_sofa_string: str,
        feature_structures: list[FeatureStructure]
    ) -> list[dict[str, int | str]]:
        cutting_points = {fs.begin for fs in feature_structures} | {fs.end for fs in feature_structures}
        cutting_points |= {i + 1 for i, ch in enumerate(cas_sofa_string) if ch.isspace()}

        points = sorted(cutting_points | {0, len(cas_sofa_string)})

        tokens: list[dict[str, int | str]] = []
        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            if start == end:
                continue # skip empty tokens
            tokens.append({"start": start, "end": end, "text": cas_sofa_string[start:end]})

        return tokens

    def _create_spans(
        self,
        cas: Cas,
        cas_sofa_tokens: list[dict[str, int|str|None]],
        annotation_type: str,
    ) -> list[dict[str, int|str|None]]:

        start_idx = {tok["start"]: i for i, tok in enumerate(cas_sofa_tokens)}
        end_idx_next = {tok["end"]: i + 1 for i, tok in enumerate(cas_sofa_tokens)}  # exklusives Ende

        spans: list[dict[str, int|str|None]] = []
        for fs in cas.select(annotation_type):
            try:
                start = start_idx[fs.begin]
                end = end_idx_next[fs.end]
            except KeyError as e:
                # Grenzen nicht exakt auf Tokens -> lieber explizit fehlschlagen
                raise VisualizerException(
                    f"Annotation [{fs.begin}, {fs.end}] of type {annotation_type} "
                    f"does not align with token boundaries."
                ) from e

            spans.append(
                {
                    "start": fs.begin,
                    "end": fs.end,
                    "start_token": start,
                    "end_token": end,
                    "label": self.get_label(fs, annotation_type),
                }
            )
        return spans

    def _parse_spans(self, cas: Cas) -> str:  # see parse_ents spaCy/spacy/displacy/__init__.py
        selected_annotations = [item for typeclass in self.type_list for item in cas.select(typeclass)]
        tmp_tokens = self._split_text_for_spans(cas.sofa_string, selected_annotations)
        tmp_token_texts = [_["text"] for _ in sorted(tmp_tokens, key=lambda t: t["start"])]

        tmp_spans = []
        labels_to_colors = dict()
        for annotation_type in self.type_list:
            for tmp_span in self._create_spans(cas=cas, cas_sofa_tokens=tmp_tokens, annotation_type=annotation_type):
                label = tmp_span["label"]
                color = self._get_color(annotation_type, label)
                if color is not None:
                    # remove spans without a color from list
                    labels_to_colors[label] = color
                    tmp_spans.append(tmp_span)
        tmp_spans.sort(key=lambda x: x["start"])
        return SpanRenderer({"colors": labels_to_colors}).render_spans(tmp_token_texts, tmp_spans, "")

class SpacyDependencyVisualizer(Visualizer):

    # TODO not sure that having defaults makes sense here. It is even a mix of DKPro and DAKODA types
    # perhaps these should be removed and the user has to provide them explicitly
    # 
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
                  options: Optional[Dict[str, Any]] = None,
                  page: bool = False,
                  start: int = 0,
                  end: int = -1,
                  output_format: OutputFormat = 'html',
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
        self._options = options or {}
        self._output_format = output_format
        self._page = page
        self._span_range = [start, end]
        if end > -1 and start > end:
            raise VisualizerException(f'Given span range [start={start}, end={end}] is not valid.')
        
        parsed = []
        renderer = DependencyRenderer(options=self._options)
        span_start = self._span_range[0]
        span_end = self._span_range[1]
        if span_end == -1:
            span_end = len(cas.sofa_string)
        for item in cas.select(self._span_type):
            if self._span_range is None or (item.begin >= span_start and item.end <= span_end):
                struct = self._dep_to_dict(cas=cas, covered=item)
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

    def _dep_to_dict(self, cas: Cas, covered: FeatureStructure):

        covered_pos = list(cas.select_covered(self._pos_type, covering_annotation=covered))
        offset_to_index = {p.begin: i for i, p in enumerate(covered_pos)}

        words = [
            {
                'text': p.get_covered_text(),
                'tag': p.PosValue
            }
            for p in covered_pos
        ]

        cbegin, cend = covered.begin, covered.end
        arcs = []
        for d in cas.select(self._dep_type):
            gb = d.Governor.begin
            db = d.Dependent.begin
            if (cbegin <= gb < cend) and (cbegin <= db < cend):
                if gb in offset_to_index and db in offset_to_index:
                    start_idx = offset_to_index[gb]
                    end_idx = offset_to_index[db]
                    arcs.append({
                        'start': start_idx,
                        'end': end_idx,
                        'label': d.DependencyType,
                        'dir': 'right' if gb < db else 'left',
                    })

        # ensure that start is always smaller than end
        # i.e. direction is only encoded in 'dir' field
        for arc in arcs:
            if arc['start'] > arc['end']:
                arc['start'], arc['end'] = arc['end'], arc['start']

        # remove root (i.e. keep everything except root where start == end)
        arcs = [arc for arc in arcs if arc['start'] != arc['end']]

        return {"words": words, "arcs": arcs}