import json
import pytest

from cassis import TypeSystem
from cas_visualizer.visualizer import (
    SpacySpanVisualizer,
    TableVisualizer,
    SpacyDependencyVisualizer,
    VisualizerException,
)

from tests.fixtures import *

T_ENT = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"


# ------------------------------
# SpanVisualizer: initialization
# ------------------------------

def test_span_visualizer_init(typesystem):
    vis = SpacySpanVisualizer(ts=typesystem)
    assert vis.selected_span_type == SpacySpanVisualizer.UNDERLINE
    assert vis._ts is not None
    assert isinstance(vis._ts, TypeSystem)


def test_span_visualizer_init_with_span_type(typesystem):
    span_type = SpacySpanVisualizer.HIGHLIGHT
    vis = SpacySpanVisualizer(ts=typesystem, span_type=span_type)
    assert vis.selected_span_type == span_type


def test_span_visualizer_init_with_span_type_error(typesystem):
    span_type = "UNKNOWN"
    with pytest.raises(VisualizerException):
        SpacySpanVisualizer(ts=typesystem, span_type=span_type)
# ------------------------------
# SpanVisualizer: HIGHLIGHT mode
# ------------------------------

def test_span_visualizer_highlight_no_overlap(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    sv.add_type(T_ENT, feature="value")  # assigns default color

    html = sv.visualize(cas_single_sentence, output_format='html')
    # Should contain labeled entities and their surface text
    assert "saw" in html
    assert "dog" in html
    # labels appear: uses feature value as label (value)
    assert "VERB" in html
    assert "ANIMAL" in html


def test_span_visualizer_highlight_overlap_raises(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    sv.add_type(T_ENT, feature="value")

    # Add overlapping entities: "saw a" [2,7] and "a dog" [6,11]
    ENT_T = typesystem.get_type(T_ENT)
    cas = cas_single_sentence
    cas.add(ENT_T(begin=2, end=7, value="PHRASE1"))
    cas.add(ENT_T(begin=6, end=11, value="PHRASE2"))

    with pytest.raises(VisualizerException):
        _ = sv.visualize(cas, output_format='html')

    # If overlap is allowed, no exception
    sv.allow_highlight_overlap = True
    html = sv.visualize(cas, output_format='html')
    assert "PHRASE1" in html and "PHRASE2" in html


def test_span_visualizer_highlight_start_end_filtering_spec(typesystem, cas_single_sentence):
    # Entities in cas_single_sentence: "saw" [2,5] -> VERB, "dog" [8,11] -> ANIMAL
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    sv.add_type(T_ENT, feature="value")

    # Build spec for a range including only "dog"
    spec = sv.build(cas_single_sentence, start=6, end=12)

    # Only one entity should be highlighted: the "dog" span
    assert "ents" in spec
    assert len(spec["ents"]) == 1

    ent = spec["ents"][0]
    assert ent["start"] == 8 and ent["end"] == 11
    assert ent["label"] == "ANIMAL"

    # Render reflects the same: ANIMAL label present, VERB omitted
    html = sv.render(spec, output_format='html')
    assert "ANIMAL" in html
    assert "VERB" not in html


def test_span_visualizer_rejects_non_html(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    sv.add_type(T_ENT, feature="value")
    with pytest.raises(VisualizerException):
        _ = sv.visualize(cas_single_sentence, output_format='json')


def test_span_visualizer_underline_start_end_filtering_spec(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.UNDERLINE)
    sv.add_type(T_ENT, feature="value")

    spec = sv.build(cas_single_sentence, start=6, end=12)

    # Underline mode spec should carry 'spans'
    assert "spans" in spec
    assert len(spec["spans"]) == 1

    span = spec["spans"][0]
    assert span["start"] == 8 and span["end"] == 11
    assert span["label"] == "ANIMAL"


def test_span_visualizer_color_resolution(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    # Type-level default color (fallback)
    sv.add_type(T_ENT, feature="value", color="steelblue")
    # Label-specific color mapping for VERB -> "Verb" label
    sv.add_feature(T_ENT, feature="value", value="VERB", label="Verb", color="orangered")

    spec = sv.build(cas_single_sentence)

    # Colors map should contain both label-specific and type-default entries
    assert spec["colors"]["Verb"] == "orangered"     # label-specific color
    assert spec["colors"]["ANIMAL"] == "steelblue"   # type default color fallback


def test_span_visualizer_highlight_invalid_output_format(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    sv.add_type(T_ENT, feature="value")
    with pytest.raises(VisualizerException):
        _ = sv.visualize(cas_single_sentence, output_format='json')


def test_span_visualizer_highlight_build_render_roundtrip(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    sv.add_type(T_ENT, feature="value")

    spec = sv.build(cas_single_sentence)
    html_render = sv.render(spec, output_format='html')
    html_vis = sv.visualize(cas_single_sentence, output_format='html')
    assert isinstance(html_render, str)
    assert html_render == html_vis


def test_span_visualizer_color_resolution(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.HIGHLIGHT)
    # Type-level default color
    sv.add_type(T_ENT, feature="value", color="steelblue")
    # Label-specific color for VERB
    sv.add_feature(T_ENT, feature="value", value="VERB", label="Verb", color="orangered")

    spec = sv.build(cas_single_sentence)
    # Colors map should contain both label-specific and type-default entries
    assert spec["colors"]["Verb"] == "orangered"      # label-specific
    assert spec["colors"]["ANIMAL"] == "steelblue"    # fallback to type color


# ------------------------------
# SpanVisualizer: UNDERLINE mode
# ------------------------------

def test_span_visualizer_underline_parse_spans(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.UNDERLINE)
    sv.add_type(T_ENT, feature="value")
    html = sv.visualize(cas_single_sentence, output_format='html')
    assert "saw" in html and "dog" in html


def test_span_visualizer_underline_build_render_roundtrip(typesystem, cas_single_sentence):
    sv = SpacySpanVisualizer(typesystem, span_type=SpacySpanVisualizer.UNDERLINE)
    sv.add_type(T_ENT, feature="value")

    spec = sv.build(cas_single_sentence)
    html_render = sv.render(spec, output_format='html')
    html_vis = sv.visualize(cas_single_sentence, output_format='html')
    assert isinstance(html_render, str)
    assert html_render == html_vis


# ------------------------------
# TableVisualizer (basic)
# ------------------------------

def test_table_visualizer_visualize_uses_output_format(typesystem, cas_single_sentence):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")
    out = tv.visualize(cas_single_sentence, output_format='json')
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 2
    # content sanity
    texts = {row["text"] for row in data}
    assert texts == {"saw", "dog"}


# ------------------------------
# Dependency visualizer (spacy-style)
# ------------------------------

def test_dependency_visualizer_dep_to_dict_basic(typesystem, cas_single_sentence):
    dv = SpacyDependencyVisualizer(
        typesystem,
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    sent_type = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    sentence = next(iter(cas_single_sentence.select(sent_type)))

    struct = dv._dep_to_dict(cas=cas_single_sentence, covered=sentence)
    words = struct["words"]
    arcs = struct["arcs"]

    # Should have 4 tokens and 3 arcs (no root)
    assert [w["text"] for w in words] == ["I", "saw", "a", "dog"]
    assert len(arcs) == 3
    labels = {a["label"] for a in arcs}
    assert {"nsubj", "obj", "det"} == labels

    # Indices should be within words range
    max_index = max(max(a["start"], a["end"]) for a in arcs)
    assert max_index < len(words)


def test_dependency_visualizer_visualize_basic(typesystem, cas_single_sentence):
    dv = SpacyDependencyVisualizer(
        typesystem,
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    html = dv.visualize(cas_single_sentence, output_format='html')
    # Should contain token texts and dependency SVG
    assert "I" in html and "saw" in html and "dog" in html
    assert "<svg" in html  # displaCy renders as SVG

def test_dependency_visualizer_build_and_render(typesystem, cas_single_sentence):
    dv = SpacyDependencyVisualizer(
        typesystem,
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    spec = dv.build(cas_single_sentence)
    html = dv.render(spec, output_format='html')
    assert isinstance(html, str)
    assert "<svg" in html


def test_dependency_visualizer_dep_to_dict_indices_within_sentence(typesystem, cas_two_sentences):
    dv = SpacyDependencyVisualizer(
        typesystem,
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    sent_type = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    sentences = list(cas_two_sentences.select(sent_type))
    assert len(sentences) == 2

    # For sentence 1, arcs must be within the words of sentence 1.
    struct = dv._dep_to_dict(cas_two_sentences, covered=sentences[0])
    words = struct["words"]
    arcs = struct["arcs"]
    max_index = max(max(a["start"], a["end"]) for a in arcs)
    assert max_index < len(words)