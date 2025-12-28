from pathlib import Path
from typing import IO, Union
import io
import cassis

def ensure_typesystem(ts: Union[cassis.TypeSystem, Path, IO[bytes]]) -> cassis.TypeSystem:
    # Already a TypeSystem
    if isinstance(ts, cassis.TypeSystem):
        return ts

    # File-like
    if hasattr(ts, "read"):
        return cassis.load_typesystem(ts)

    # Path to a typesystem file
    if isinstance(ts, Path):
        with ts.open("rb") as f:
            return cassis.load_typesystem(f)

    raise TypeError(f"Unsupported type for typesystem: {type(ts).__name__}")


def ensure_cas(
    cas: Union[cassis.Cas, str, bytes, Path, IO[bytes]],
    typesystem: Union[cassis.TypeSystem, Path, IO[bytes]],
    *,
    lenient: bool = False,
    trusted: bool = False,
) -> cassis.Cas:
    # Already a CAS
    if isinstance(cas, cassis.Cas):
        # Optional: check instance identity of the TypeSystem if provided
        ts = ensure_typesystem(typesystem)
        if cas.typesystem is not ts:
            raise ValueError("CAS is already loaded with a different TypeSystem instance.")
        return cas

    ts = ensure_typesystem(typesystem)

    # Raw bytes (not directly supported by cassis) -> wrap
    if isinstance(cas, (bytes, bytearray)):
        return cassis.load_cas_from_xmi(io.BytesIO(cas), typesystem=ts, lenient=lenient, trusted=trusted)

    # XML string content (cassis treats str as XMI)
    if isinstance(cas, str):
        return cassis.load_cas_from_xmi(cas, typesystem=ts, lenient=lenient, trusted=trusted)

    # Path to an XMI file
    if isinstance(cas, Path):
        return cassis.load_cas_from_xmi(cas, typesystem=ts, lenient=lenient, trusted=trusted)

    # File-like
    if hasattr(cas, "read"):
        return cassis.load_cas_from_xmi(cas, typesystem=ts, lenient=lenient, trusted=trusted)

    raise TypeError(
        f"Unsupported type for 'cas': {type(cas).__name__}. "
        f"Expected cassis.Cas, str (XMI), bytes, Path, or file-like."
    )

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

