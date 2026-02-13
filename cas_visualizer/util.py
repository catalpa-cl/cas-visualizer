from cassis.typesystem import TypeSystem
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

def resolve_annotation(annotation_path: str, feature_separator='/') -> tuple[str, str]:
    if feature_separator == '.':
        raise ValueError('Feature separator must not be "."')

    split = annotation_path.split(feature_separator)

    if len(split) > 2:
        raise ValueError(f'Annotation Path is ill defined, as it contains multiple features, seperated by {feature_separator}')

    # no feature in annotation path
    if len(split) == 1:
        return split[0], ''

    type_path, feature_path = split

    return type_path, feature_path

def suggest_type_name(ts: TypeSystem, query: str) -> str | None:
    """
    Inspect the TypeSystem and return the best-matching type name for `query`.

    Ties: prefer shorter last segment, then shorter full path, then lexicographic order.

    Returns:
      - Best-matching fully qualified type name, or None if the TypeSystem has no types.
    """
    q = (query or "").strip()
    if not q:
        return None

    names = [t.name if hasattr(t, "name") else str(t) for t in ts.get_types()]
    if not names:
        return None

    # If the query looks fully qualified and exists, return it directly
    if "." in q:
        try:
            ts.get_type(q)  # will raise if not present
            return q
        except Exception:
            pass  # fall through to heuristic matching

    q_lower = q.lower()

    def score(tname: str) -> tuple[int, int, int, str]:
        last = tname.rsplit(".", 1)[-1]
        last_lower = last.lower()

        # 0: full exact
        if tname == q:
            return (0, len(last), len(tname), tname)

        # 1: last exact, case-sensitive 
        if last == q:
            return (1, len(last), len(tname), tname)

        # 2: last exact, case-insensitive
        if last_lower == q_lower:
            return (2, len(last), len(tname), tname)

        # 3: last endswith(query), case-insensitive
        if last_lower.endswith(q_lower):
            return (3, len(last), len(tname), tname)

        # 4: last contains(query) but NOT at start or end!
        if (
            q_lower in last_lower 
            and not last_lower.endswith(q_lower) 
            and not last_lower.startswith(q_lower)
        ):
            return (4, len(last), len(tname), tname)

        # Kein Match für query NUR im Package-Namen!
        return (9_999, len(last), len(tname), tname)

    best = min(names, key=score)
    best_score = score(best)[0]

    # If even the best score is the default "no match" score, signal no suggestion
    if best_score >= 9_999:
        return None
    return best