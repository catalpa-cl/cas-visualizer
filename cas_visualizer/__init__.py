"""
cas_visualizer: Visualize CAS (Common Annotation System) annotations.

A Python library for visualizing annotations from the Common Annotation System (CAS)
in various formats including Spacy-style HTML spans, dependency trees, tables, heatmaps, and DOCX.

Quick start:
    >>> from cas_visualizer import SpacySpanVisualizer
    >>> vis = SpacySpanVisualizer('path/to/typesystem.xml')
    >>> html = vis.visualize(cas_object)
    >>> print(html)

Main visualizers:
    - SpacySpanVisualizer: HTML span visualization using spaCy's displaCy
    - DocxSpanVisualizer: DOCX span visualization with colored annotations
    - TableVisualizer: CSV/HTML table of annotations
    - UdapiDependencyVisualizer: UDPipe/UDAPI dependency tree format
    - SpacyDependencyVisualizer: spaCy dependency tree HTML visualization
    - HeatmapVisualizer: Matplotlib heatmap of annotation density
"""

from cas_visualizer._base import Visualizer, VisualizerException, TypeConfig
from cas_visualizer.span import SpacySpanVisualizer, DocxSpanVisualizer
from cas_visualizer.table import TableVisualizer
from cas_visualizer.dependency import UdapiDependencyVisualizer, SpacyDependencyVisualizer
from cas_visualizer.heatmap import HeatmapVisualizer

__version__ = "0.2.0"

__all__ = [
    "Visualizer",
    "VisualizerException",
    "TypeConfig",
    "SpacySpanVisualizer",
    "DocxSpanVisualizer",
    "TableVisualizer",
    "UdapiDependencyVisualizer",
    "SpacyDependencyVisualizer",
    "HeatmapVisualizer",
]
