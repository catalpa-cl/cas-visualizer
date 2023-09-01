import typing
import cassis
import pathlib
import functools

def load_cas(cas, typesystem) -> cassis.Cas:
    typesystem = load_typesystem(typesystem)

    cas_load_fn = functools.partial(cas_from_string, typesystem=typesystem)
    type_map = {
        str: cas_load_fn,
        pathlib.Path: cas_load_fn,
        cassis.Cas: identity
    }
    return map_from_type(cas, type_map)

def load_typesystem(typesystem: typing.Union[cassis.TypeSystem, str]) -> cassis.TypeSystem:
    def load_typesystem_from_file(path):
        with open(path, 'rb') as f:
            return cassis.load_typesystem(f)

    ts_typemap = {
        pathlib.Path: load_typesystem_from_file,
        str: load_typesystem_from_file,
        cassis.TypeSystem: identity
    }

    return map_from_type(typesystem, ts_typemap)

def identity(x):
    return x


def map_from_type(x, type_to_mapper: dict):
    T = type(x)
    mapping_fn = type_to_mapper.get(T)

    if mapping_fn is None:
        raise ValueError

    return mapping_fn(x)

def cas_from_string(cas: str, typesystem) -> cassis.Cas:
    with open(cas, 'rb') as f:
        return cassis.load_cas_from_xmi(f, typesystem)

def resolve_annotation(annotation_path: str, feature_seperator='/') -> tuple[str, str]:
    if feature_seperator == '.':
        raise ValueError('Feature separator must not be "."')

    split = annotation_path.split(feature_seperator)

    if len(split) > 2:
        raise ValueError(f'Annotation Path is ill defined, as it contains multiple features, seperated by {feature_seperator}')

    # no feature in annotation path
    if len(split) == 1:
        return split[0], ''

    type_path, feature_path = split

    return type_path, feature_path