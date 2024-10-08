import abc
import copy
import dataclasses
import functools
import typing

import cassis
import numpy as np
import pandas as pd

from cas_visualizer import util as util
from cas_visualizer import spacy_helper as sh

class AnnoObject:
    def __init__(self, anno_begin, anno_end, anno_type, anno_text):
        self.anno_begin = anno_begin
        self.anno_end = anno_end
        self.anno_type = anno_type
        self.anno_text = anno_text

class VisualisationConfig:
    def __init__(self,
                 annotation: str,
                 tooltip: str = '',
                 subscript: str = ''
    ):
        self.annotation = annotation
        self.tooltip = tooltip
        self.subscript = subscript
        type_path, feature_path = util.resolve_annotation(annotation)
        self.type_path = type_path
        self.feature_path = feature_path

    @classmethod
    def from_any(cls, config):
        type_dict = {
            cls: cls.from_config,
            str: cls.from_string,
            dict: cls.from_dict
        }
        return util.map_from_type(config, type_dict)

    @classmethod
    def from_config(cls, config):
        return copy.deepcopy(config)

    @classmethod
    def from_dict(cls, config: dict):
        return cls(
            config['annotation'],
            config['tooltip'],
            config['subscript']
        )

    @classmethod
    def from_string(cls, annotation: str):
        return cls(
            annotation,
            '',
            ''
        )

class Visualiser(abc.ABC):
    def __init__(
            self,
            cas: cassis.Cas,
            visualisation_configs: typing.Iterable[VisualisationConfig]
    ):
        self.cas = cas
        self.visualisation_configs = visualisation_configs if visualisation_configs is not None else []

    def __call__(self, streamlit_context=None, *args, **kwargs):
        self.visualise(streamlit_context)

    @functools.cached_property
    def entities(self) -> typing.List[typing.List]:
        """Returns a list of entities to be visualised. One list containing types is returned for each configuration."""
        entities = []
        for cfg in self.visualisation_configs:
            entities.append(list(self.cas.select(cfg.type_path)))
        return entities

    @functools.cached_property
    def unique_entity_values(self):
        """Returns unique entity values to be used for visualisation, one list of values per visualisation config."""
        entities = self.entities
        values = []
        for entity_list, cfg in zip(entities, self.visualisation_configs):
            vs = [entity.get(cfg.feature_path) for entity in entity_list]
            values.append(np.unique(vs).tolist())
        return values

    def visualise(self):
        return self.render_visualisation()

    @abc.abstractmethod
    def render_visualisation(self):
        """Generates the visualisation based on the provided configuration."""
        raise NotImplementedError


class TableVisualiser(Visualiser):
    def render_visualisation(self):
        records = []
        for entity_list, cfg in zip(self.entities, self.visualisation_configs):
            for entity in entity_list:
                records.append({
                    'text': entity.get_covered_text(),
                    'feature': cfg.feature_path,
                    'value': entity.get(cfg.feature_path),
                    'begin': entity.begin,
                    'end': entity.end,
                })

        df = pd.DataFrame.from_records(records).sort_values(by=['begin', 'end'])
        return df
    
class SpanVisualiser(Visualiser):
    def render_visualisation(self):

        cas_text = self.cas.sofa_string

        allTypes = []
        unsortedAnnos = []
        for cfg in self.visualisation_configs:
            typePath, featPath = util.resolve_annotation(cfg.annotation)

            for item in self.cas.select(typePath):
                annoobject = AnnoObject(item.begin, item.end, getattr(item, featPath), item.get_covered_text())
                unsortedAnnos.append(annoobject)

                # get all possible Types
                if getattr(item, featPath) not in allTypes:
                    allTypes.append(getattr(item, featPath))

        sortedAnnos = sorted(unsortedAnnos, key=lambda x: x.anno_begin, reverse=False)

        # currently configured to show all types at visualization time (TODO: make configurable)
        selectedTypes = allTypes

        color_scheme1 = ["skyblue", "orangered", "orange", "plum", "palegreen", "mediumseagreen", "lightseagreen",
                        "steelblue", "navajowhite", "mediumpurple", "rosybrown", "silver", "gray",
                        "paleturquoise", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]

        colorMapping = {}  # for the type-color display above the text

        # assign each type a unique color
        for my_type in allTypes:
            colorMapping[my_type] = color_scheme1[allTypes.index(my_type)]

        #create legend
        legend = ''
        for type in selectedTypes:
            legend = legend + addAnnotationVisHTML(type, colorMapping[type])

        revSortAnnos = sorted(sortedAnnos, key=lambda x: x.anno_begin, reverse=True)
        textToPrint = cas_text

        for anno in revSortAnnos:
            if anno.anno_type in selectedTypes:
                htmlend = "</span> "
                htmlstart = "<span style=\"border-radius: 25px; padding-left:8px; padding-right:8px; background-color: " + \
                            str(colorMapping[anno.anno_type]) + "\">"

                textToPrint = textToPrint[:anno.anno_end] + htmlend + textToPrint[anno.anno_end:]
                textToPrint = textToPrint[:anno.anno_begin] + htmlstart + textToPrint[anno.anno_begin:]


        textToPrint = textToPrint.replace('\n', '<br>')

        return (legend, textToPrint)
        

# quick method to wrap the html part around a token (assign background color)
# can be modified like normal HTML
def addAnnotationVisHTML(text, color):
    if color != 'noColor':
        return "<span style=\"border-radius: 25px; padding-left:10px; padding-right:10px; background-color: " + \
            str(color) + "\">" + str(text) + "</span> "
    else:
        return text + ' '

class VisualiserException(Exception):
    pass

class SpacySpanVisualiser(Visualiser):
    SPAN_STYLE_HIGHLIGHTING = 'SPAN_STYLE_HIGHLIGHTING'
    SPAN_STYLE_UNDERLINING = 'SPAN_STYLE_UNDERLINING'

    _span_type = ""
    _selected_annotations_to_types = dict()
    _annotations_to_colors = dict()
    _allow_highlighting_overlap = False

    def set_annotations_to_colors(self, annotations_to_colors):
        self._annotations_to_colors = annotations_to_colors

    def set_selected_annotations_to_types(self, selected_annotations_to_types):
        self._selected_annotations_to_types = selected_annotations_to_types

    def set_span_type(self, span_type):
        self._span_type = span_type

    def set_allow_highlighting_overlap(self, allow_highlighting_overlap):
        self._allow_highlighting_overlap = allow_highlighting_overlap

    def render_visualisation(self):
        show_overlap = False
        if self._span_type == SpacySpanVisualiser.SPAN_STYLE_HIGHLIGHTING:
            html, has_overlap = sh.parse_ents(self.cas, self._selected_annotations_to_types, self._annotations_to_colors)
            if has_overlap and not self._allow_highlighting_overlap:
                raise VisualiserException('The highlighted annotations are overlapping. Please choose a different set of annotations for this display style or switch to a different style.')
        elif self._span_type == SpacySpanVisualiser.SPAN_STYLE_UNDERLINING:
            html = sh.parse_spans(self.cas, self._selected_annotations_to_types, self._annotations_to_colors)
        return html
