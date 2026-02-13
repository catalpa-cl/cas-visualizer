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

# ------------------------------
# Dependency visualizer (spacy-style)
# ------------------------------

def test_dependency_visualizer_dep_to_dict_basic(typesystem, cas_single_sentence):
    from cas_visualizer.visualizer import DependencyVisualizerConfig
    config = DependencyVisualizerConfig(
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    dv = SpacyDependencyVisualizer(typesystem, config)
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
    from cas_visualizer.visualizer import DependencyVisualizerConfig
    config = DependencyVisualizerConfig(
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    dv = SpacyDependencyVisualizer(typesystem, config)
    html = dv.visualize(cas_single_sentence, output_format='html')
    # Should contain token texts and dependency SVG
    assert "I" in html and "saw" in html and "dog" in html
    assert "<svg" in html  # displaCy renders as SVG

def test_dependency_visualizer_build_and_render(typesystem, cas_single_sentence):
    from cas_visualizer.visualizer import DependencyVisualizerConfig
    config = DependencyVisualizerConfig(
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    dv = SpacyDependencyVisualizer(typesystem, config)
    spec = dv.build(cas_single_sentence)
    html = dv.render(spec, output_format='html')
    assert isinstance(html, str)
    assert "<svg" in html


def test_dependency_visualizer_dep_to_dict_indices_within_sentence(typesystem, cas_two_sentences):
    from cas_visualizer.visualizer import DependencyVisualizerConfig
    config = DependencyVisualizerConfig(
        dep_type="org.dakoda.syntax.UDependency",
        pos_type="de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        span_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    dv = SpacyDependencyVisualizer(typesystem, config)
    sent_type = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    sentences = list(cas_two_sentences.select(sent_type))
    assert len(sentences) == 2

    # For sentence 1, arcs must be within the words of sentence 1.
    struct = dv._dep_to_dict(cas_two_sentences, covered=sentences[0])
    words = struct["words"]
    arcs = struct["arcs"]
    max_index = max(max(a["start"], a["end"]) for a in arcs)
    assert max_index < len(words)