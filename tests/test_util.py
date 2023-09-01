import pytest

from cas_visualizer.util import resolve_annotation

def test_resolve_annotation():

    only_type = 'catalpa.xyz'
    assert list(resolve_annotation(only_type)) == ['catalpa.xyz', '']

    feature_path = 'catalpa.xyz/feature'
    assert list(resolve_annotation(feature_path)) == ['catalpa.xyz', 'feature']
    
    non_standard_separator = 'catalpa.xyz#feature'
    assert list(resolve_annotation(non_standard_separator, '#')) == ['catalpa.xyz', 'feature']

    ill_formed = 'catalpa.xyz//feature'
    with pytest.raises(ValueError):
        resolve_annotation(ill_formed)

    multiple_features = 'catalpa.xyz/feature/feature'
    with pytest.raises(ValueError):
        resolve_annotation(multiple_features)