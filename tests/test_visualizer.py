from cas_visualizer.visualizer import SpanVisualizer, VisualizerException
from cassis import TypeSystem

import pytest

cas = 'data/hagen.txt.xmi'
ts = 'data/TypeSystem.xml'

def test_span_visualizer_init():
    vis = SpanVisualizer(ts)
    assert vis.selected_span_type == SpanVisualizer.UNDERLINE
    assert vis._ts is not None
    assert isinstance(vis._ts, TypeSystem)

def test_span_visualizer_init_with_types():
    type_name = 'NamedEntity'
    vis = SpanVisualizer(ts, types=[type_name])
    assert type_name in vis._types

def test_span_visualizer_init_with_span_type():
    span_type = "HIGHLIGHT"
    vis = SpanVisualizer(ts, span_type=span_type)
    assert vis.selected_span_type == span_type

def test_span_visualizer_init_with_span_type_error():
    span_type = "UNKNOWN"
    with pytest.raises(VisualizerException):
        SpanVisualizer(ts, span_type=span_type)



