import pytest
from pathlib import Path
from cassis import Cas

from cas_visualizer.heatmap import HeatmapVisualizer, VisualizerException

from tests.fixtures import *

T_ENT = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
T_SENT = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
T_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"


def test_heatmap_invalid_hex_color_initialization(typesystem):
    """Test that invalid hex colors are handled gracefully during initialization."""
    # The key test: verify that invalid hex doesn't crash the constructor
    # due to try/except in _hex_complement
    hm = HeatmapVisualizer(
        typesystem,
        token_type=T_TOKEN,
        sentence_type=T_SENT,
        color_hex="#ZZZZZZ",  # Invalid hex
    )
    hm.add_type(T_ENT)
    # Should not crash during initialization
    assert hm is not None
    assert hm._color_hex == "#ZZZZZZ"


def test_heatmap_empty_cas_strict(typesystem):
    """Test that empty CAS raises exception in strict mode."""
    hm = HeatmapVisualizer(
        typesystem,
        token_type=T_TOKEN,
        sentence_type=T_SENT,
        strict=True,
    )
    
    empty_cas = Cas(typesystem=typesystem)
    empty_cas.sofa_string = ""
    
    with pytest.raises(VisualizerException):
        hm.build(empty_cas)


def test_heatmap_no_types_configured(typesystem):
    """Test that building without configured types raises exception."""
    hm = HeatmapVisualizer(
        typesystem,
        token_type=T_TOKEN,
        sentence_type=T_SENT,
        strict=False,
    )
    # Don't add any types
    
    empty_cas = Cas(typesystem=typesystem)
    empty_cas.sofa_string = ""
    
    with pytest.raises(VisualizerException):
        hm.build(empty_cas)


def test_heatmap_hex_complement_valid(typesystem):
    """Test that _hex_complement works with valid hex colors."""
    result = HeatmapVisualizer._hex_complement("#FF0000")
    assert result == "#00FFFF"  # Red -> Cyan
    
    result = HeatmapVisualizer._hex_complement("#FFFFFF")
    assert result == "#000000"  # White -> Black


def test_heatmap_hex_complement_invalid(typesystem):
    """Test that _hex_complement handles invalid hex gracefully."""
    result = HeatmapVisualizer._hex_complement("#ZZZZZZ")
    assert result == "#00B2FF"  # Returns default
    
    result = HeatmapVisualizer._hex_complement("notahex")
    assert result == "#00B2FF"  # Returns default


def test_heatmap_hex_to_rgb_valid(typesystem):
    """Test that _hex_to_rgb works with valid colors."""
    r, g, b = HeatmapVisualizer._hex_to_rgb("#FF0000")
    assert (r, g, b) == (255, 0, 0)
    
    r, g, b = HeatmapVisualizer._hex_to_rgb("#00FF00")
    assert (r, g, b) == (0, 255, 0)


def test_heatmap_hex_to_rgb_invalid(typesystem):
    """Test that _hex_to_rgb handles invalid hex gracefully."""
    r, g, b = HeatmapVisualizer._hex_to_rgb("#ZZZZZZ")
    assert (r, g, b) == (255, 69, 0)  # Returns default orangered
    
    r, g, b = HeatmapVisualizer._hex_to_rgb("short")
    assert (r, g, b) == (255, 69, 0)  # Returns default orangered


def test_heatmap_initialization_with_custom_colors(typesystem):
    """Test heatmap initialization with custom color settings."""
    hm = HeatmapVisualizer(
        typesystem,
        token_type=T_TOKEN,
        sentence_type=T_SENT,
        color_hex="#FF5500",
        unannotated_hex="#CCCCCC",
        background_hex="#FFFFFF",
    )
    assert hm._color_hex == "#FF5500"
    assert hm._unannotated_hex == "#CCCCCC"
    assert hm._background_hex == "#FFFFFF"


def test_heatmap_invalid_output_format_raises(typesystem):
    """Test that invalid output format is caught properly."""
    hm = HeatmapVisualizer(
        typesystem,
        token_type=T_TOKEN,
        sentence_type=T_SENT,
    )
    hm.add_type(T_ENT)
    
    # Create a minimal valid spec manually (since fixtures don't have tokens)
    spec = {"format": "html"}  # Dummy spec
    
    # Try to render with invalid format
    with pytest.raises(VisualizerException):
        hm.render(spec, output_format="invalid_format")

