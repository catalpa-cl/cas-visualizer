import typing
from pathlib import Path

import cassis
from util import load_cas

from visualizer import TableVisualiser, SpanVisualiser, VisualisationConfig

def highlight(cas: typing.Union[cassis.Cas, str, Path],
              typesystem: typing.Union[cassis.TypeSystem, str, Path],
              config: typing.Iterable[typing.Union[str, dict, VisualisationConfig]],
              context=None):
    cas = load_cas(cas, typesystem)
    # for now like this, but actually conf is passed as parameter above
    conf = VisualisationConfig('')
    visualiser = SpanVisualiser(cas, [conf])
    return visualiser(context)

def table(cas: typing.Union[cassis.Cas, str, Path],
          typesystem: typing.Union[cassis.TypeSystem, str, Path],
          config: typing.Iterable[typing.Union[str, dict, VisualisationConfig]],
          context=None):
    # TODO: Resolve CAS, Typesystem and Config automatically
    cas = load_cas(cas, typesystem)
    # for now like this, but actually conf is passed as parameter above
    conf = VisualisationConfig('')
    visualiser = TableVisualiser(cas, [conf])
    return visualiser(context)