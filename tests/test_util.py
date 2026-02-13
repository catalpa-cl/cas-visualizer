import pytest

from tests.fixtures import *
from cassis.typesystem import TypeSystem
from cas_visualizer.util import resolve_annotation, suggest_type_name

def test_resolve_annotation():

    only_type = 'catalpa.xyz'
    assert list(resolve_annotation(only_type)) == ['catalpa.xyz', '']

    feature_path = 'catalpa.xyz/feature'
    assert list(resolve_annotation(feature_path)) == ['catalpa.xyz', 'feature']
    
    non_standard_separator = 'catalpa.xyz#feature'
    assert list(resolve_annotation(non_standard_separator, feature_separator='#')) == ['catalpa.xyz', 'feature']

    ill_formed = 'catalpa.xyz//feature'
    with pytest.raises(ValueError):
        resolve_annotation(ill_formed)

    multiple_features = 'catalpa.xyz/feature/feature'
    with pytest.raises(ValueError):
        resolve_annotation(multiple_features)

def test_suggest_type_name(typesystem: TypeSystem):

    # 0: Full exact match
    assert suggest_type_name(typesystem, "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence") == \
           "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

    # 1: Last segment exact (case-sensitive)
    assert suggest_type_name(typesystem, "Sentence") == \
           "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

    # 2: Last segment exact (case-insensitive)
    assert suggest_type_name(typesystem, "sentence") == \
           "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

    # 3: Last segment endswith(query), case-insensitive
    assert suggest_type_name(typesystem, "Dependency") == \
           "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"

    assert suggest_type_name(typesystem, "POS") == \
           "custom.Stanford.POS"

    # No good matches
    assert suggest_type_name(typesystem, "pos.POS") is None
    assert suggest_type_name(typesystem, "syntax") is None
    assert suggest_type_name(typesystem, "api") is None
    assert suggest_type_name(typesystem, "dkpro") is None