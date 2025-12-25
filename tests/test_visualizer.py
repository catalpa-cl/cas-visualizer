import pytest

from cas_visualizer.visualizer import SpanVisualizer, VisualizerException, TableVisualizer, SpacyDependencyVisualizer
from cassis import TypeSystem

from tests.fixtures import *

cas = 'data/hagen.txt.xmi'
TS = 'data/TypeSystem.xml'

def test_span_visualizer_init():
    vis = SpanVisualizer(ts=TS)
    assert vis.selected_span_type == SpanVisualizer.UNDERLINE
    assert vis._ts is not None
    assert isinstance(vis._ts, TypeSystem)

def test_span_visualizer_init_with_types():
    type_name = 'NamedEntity'
    vis = SpanVisualizer(ts=TS, types=[type_name])
    assert type_name in vis._types

def test_span_visualizer_init_with_span_type():
    span_type = "HIGHLIGHT"
    vis = SpanVisualizer(ts=TS, span_type=span_type)
    assert vis.selected_span_type == span_type

def test_span_visualizer_init_with_span_type_error():
    span_type = "UNKNOWN"
    with pytest.raises(VisualizerException):
        SpanVisualizer(ts=TS, span_type=span_type)


# ------------------------------
# TableVisualizer
# ------------------------------

def test_table_visualizer_dataframe_contents(typesystem, cas_single_sentence):
    tv = TableVisualizer(typesystem)
    tv.add_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", feature="value")  # ensure feature is set

    df = tv.visualize(cas_single_sentence)

    # Required columns present
    assert set(["text", "feature", "value", "begin", "end"]).issubset(df.columns)

    # Expect two rows, ordered by begin
    texts = list(df.sort_values(["begin", "end"])["text"])
    assert texts == ["saw", "dog"]

    # Check values mapped from feature
    recs = {(row["text"], row["value"]) for _, row in df.iterrows()}
    assert ("saw", "VERB") in recs
    assert ("dog", "ANIMAL") in recs


# ------------------------------
# SpanVisualizer
# ------------------------------

def test_span_visualizer_highlight_no_overlap(typesystem, cas_single_sentence):
    sv = SpanVisualizer(typesystem, span_type=SpanVisualizer.HIGHLIGHT)
    sv.add_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", feature="value")  # assigns default color

    html = sv._parse_ents(cas_single_sentence)
    # Should contain labeled entities and their surface text
    assert "saw" in html
    assert "dog" in html
    # labels appear: uses feature value as label (value)
    assert "VERB" in html
    assert "ANIMAL" in html


def test_span_visualizer_highlight_overlap_raises(typesystem, cas_single_sentence):
    sv = SpanVisualizer(typesystem, span_type=SpanVisualizer.HIGHLIGHT)
    sv.add_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", feature="value")

    # Add overlapping entities: "saw a" [2,7] and "a dog" [6,11]
    ENT_T = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity")
    cas = cas_single_sentence
    cas.add(ENT_T(begin=2, end=7, value="PHRASE1"))
    cas.add(ENT_T(begin=6, end=11, value="PHRASE2"))

    with pytest.raises(VisualizerException):
        _ = sv._parse_ents(cas)

    # If overlap is allowed, no exception
    sv.allow_highlight_overlap = True
    html = sv._parse_ents(cas)
    assert "PHRASE1" in html and "PHRASE2" in html


#@pytest.mark.xfail(reason="Known bug: SpanVisualizer.parse_spans uses FS like dict; fix create_tokens first.")
def test_span_visualizer_underline_parse_spans(typesystem, cas_single_sentence):
    sv = SpanVisualizer(typesystem, span_type=SpanVisualizer.UNDERLINE)
    sv.add_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", feature="value")
    html = sv._parse_spans(cas_single_sentence)
    assert "saw" in html and "dog" in html





# ------------------------------
# DependencyVisualizer
# ------------------------------

def test_dependency_visualizer_dep_to_dict_basic(typesystem, cas_single_sentence):
    dv = SpacyDependencyVisualizer(typesystem)
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
    dv = SpacyDependencyVisualizer(typesystem)
    html = dv.visualize(cas_single_sentence)
    # Should contain token texts and dependency SVG
    assert "I" in html and "saw" in html and "dog" in html
    assert "<svg" in html  # displaCy renders as SVG


@pytest.mark.xfail(reason="Known bug: dep_to_dict does not restrict arcs to the covered span and uses global indices.")
def test_dependency_visualizer_dep_to_dict_indices_outside_sentence(typesystem, cas_two_sentences):
    dv = SpacyDependencyVisualizer(typesystem)
    sent_type = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    sentences = list(cas_two_sentences.select(sent_type))
    assert len(sentences) == 2

    # For sentence 1, arcs must be within the words of sentence 1.
    struct = dv._dep_to_dict(cas_two_sentences, covered=sentences[0])
    words = struct["words"]
    arcs = struct["arcs"]
    max_index = max(max(a["start"], a["end"]) for a in arcs)
    # This assertion currently fails because arcs include dependencies from sentence 2 (global index mapping).
    assert max_index < len(words)


