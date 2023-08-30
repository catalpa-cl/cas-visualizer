import abc
import copy
import dataclasses
import functools
import typing

import cassis
import numpy as np
import pandas as pd
import streamlit as st

import util as util


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

    def visualise(self, streamlit_context=None):
        """Generates the visualisation based on the provided configuration in the provided context.

        :arg streamlit_context: A streamlit context to render the visualisation in.
            Must implement the context provider protocol.
            If not provided, the global streamlit context should be used.
        """
        if streamlit_context is None: # render wherever the function is called in the global context.
            self.render_visualisation()
        else:
            with streamlit_context: # render within the given container
                self.render_visualisation()

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
        return st.table(df)
    
class SpanVisualiser(Visualiser):
    def render_visualisation(self):
    
        return st.write()

