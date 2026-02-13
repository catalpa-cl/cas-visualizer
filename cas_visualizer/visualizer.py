"""
Convenience re-export module for backwards compatibility.

Provides access to all visualizer classes without needing to know the internal module structure.
Users can import from this module or directly from cas_visualizer package.

Examples:
    From this module:
    >>> from cas_visualizer.visualizer import SpacySpanVisualizer
    
    Or directly from package:
    >>> from cas_visualizer import SpacySpanVisualizer
"""

from cas_visualizer._base import Visualizer, VisualizerException, TypeConfig
from cas_visualizer.span import SpacySpanVisualizer, DocxSpanVisualizer
from cas_visualizer.table import TableVisualizer
from cas_visualizer.dependency import UdapiDependencyVisualizer, SpacyDependencyVisualizer, DependencyVisualizerConfig, DependencyFeatureConfig
from cas_visualizer.heatmap import HeatmapVisualizer

__all__ = [
    "Visualizer",
    "VisualizerException",
    "TypeConfig",
    "SpacySpanVisualizer",
    "DocxSpanVisualizer",
    "TableVisualizer",
    "UdapiDependencyVisualizer",
    "SpacyDependencyVisualizer",
    "DependencyVisualizerConfig",
    "DependencyFeatureConfig",
    "HeatmapVisualizer",
]
