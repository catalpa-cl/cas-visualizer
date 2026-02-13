from __future__ import annotations

import abc
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Any, Iterator

from cassis import Cas, TypeSystem
from cassis.typesystem import FeatureStructure

from cas_visualizer.util import ensure_typesystem


class VisualizerException(Exception):
    """
    Domain-specific error for visualizer configuration and rendering.

    Raised when:
    - configuration is invalid (unknown type/feature, illegal span type),
    - no content can be rendered in strict mode,
    - requested output format is unsupported,
    - spec cannot be produced (e.g., misaligned annotation boundaries).
    """
    pass


# ---------- Type configuration ----------

@dataclass
class TypeConfig:
    """
    Per-type configuration used by visualizers.

    Fields:
    - name: fully qualified CAS type name (e.g., "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity")
    - feature: feature name to use for labels (e.g., "value"); if None, no label
    - default_color: fallback color for this type
    - default_label: fallback label if feature value is missing
    - value_labels: raw feature value -> display label mapping
    - label_colors: display label -> color mapping
    """
    name: str
    feature: str | None = None
    default_color: str | None = None
    default_label: str | None = None
    value_labels: dict[Any, str] = field(default_factory=dict)
    label_colors: dict[str, str] = field(default_factory=dict)

    def label_for_value(self, value: Any) -> str | None:
        """
        Resolve a display label:
        - use value_labels mapping if present,
        - else str(value) if value is not None,
        - else fallback to default_label.
        """
        mapped = self.value_labels.get(value)
        if mapped is not None:
            return mapped
        if value is not None:
            return str(value)
        return self.default_label

    def color_for_label(self, label: str | None) -> str | None:
        """
        Resolve a color:
        - use label_colors[label] if present,
        - else fallback to default_color.
        """
        if label is None:
            return self.default_color
        return self.label_colors.get(label, self.default_color)


# ---------- Base visualizer ----------

class Visualizer(abc.ABC):
    """
    Base class for CAS visualizers.

    Responsibilities:
    - hold per-type configuration (features, labels, colors),
    - provide helpers to resolve labels and colors,
    - normalize the type system argument via ensure_typesystem,
    - offer an HTML page wrapper (UTF‑8 meta) for fragments.

    Contract:
    - visualize and render return a single string, always.
    - Subclasses validate supported output_format values at runtime and raise VisualizerException otherwise.
    """
    def __init__(self, ts: str | Path | TypeSystem):
        # Registered type configs keyed by full type path
        self._type_configs: dict[str, TypeConfig] = {}

        # Default color iterator (cycled, never exhausted)
        color_list = [
            "lightgreen", "orangered", "orange", "plum", "palegreen",
            "mediumseagreen", "steelblue", "skyblue", "navajowhite",
            "mediumpurple", "rosybrown", "silver", "gray", "paleturquoise",
        ]
        self._default_colors: Iterator[str] = cycle(color_list)

        # Normalize and store the type system
        self._ts = ensure_typesystem(ts)

    @property
    def ts(self) -> TypeSystem:
        """The normalized TypeSystem used by this visualizer."""
        return self._ts

    # ----- Configuration API -----

    def add_type(
        self,
        name: str,
        feature: str | None = None,
        color: str | None = None,
        label: str | None = None,
    ) -> None:
        """
        Register or update a type configuration.

        Parameters:
        - name: fully qualified CAS type name
        - feature: feature to use for labels (optional)
        - color: default color (optional; if omitted, a new default is assigned)
        - label: default label (optional; if omitted, last path segment is used)
        """
        if not name:
            raise VisualizerException("type path cannot be empty")

        cfg = self._type_configs.get(name)
        if cfg is None:
            cfg = TypeConfig(name=name)
            self._type_configs[name] = cfg

        if feature is not None:
            # Optional: validate against TypeSystem (expensive if repeated)
            # ts_type = self._ts.get_type(name)
            # if feature not in {f.name for f in ts_type.features}:
            #     raise VisualizerException(f'Unknown feature "{feature}" for type "{name}"')
            cfg.feature = feature

        if color is not None:
            cfg.default_color = color
        elif cfg.default_color is None:
            cfg.default_color = next(self._default_colors)

        cfg.default_label = label if label is not None else (cfg.default_label or name.split(".")[-1])

    def add_feature(
        self,
        name: str,
        feature: str,
        value: Any,
        color: str | None = None,
        label: str | None = None,
    ) -> None:
        """
        Map a specific feature value to a display label and color.

        Parameters:
        - name: CAS type name to configure
        - feature: feature on the type (stored on the type config)
        - value: raw feature value to map
        - color: optional color for the display label (assigned if given; otherwise, a default is picked)
        - label: optional display label (defaults to str(value))
        """
        if not name:
            raise VisualizerException("type name cannot be empty")

        cfg = self._type_configs.get(name)
        if cfg is None:
            cfg = TypeConfig(name=name)
            self._type_configs[name] = cfg

        if not feature:
            raise VisualizerException(f"a feature for type {name} must be specified")
        cfg.feature = feature

        if value is None:
            raise VisualizerException(f"a value for feature {feature} must be specified")

        eff_label = label if label is not None else str(value)
        cfg.value_labels[value] = eff_label

        if color is not None:
            cfg.label_colors[eff_label] = color
        elif eff_label not in cfg.label_colors:
            cfg.label_colors[eff_label] = next(self._default_colors)

        if cfg.default_color is None:
            cfg.default_color = next(self._default_colors)

    def remove_type(self, type_path: str) -> None:
        """Remove a type configuration and all associated mappings."""
        if not type_path:
            raise VisualizerException("type path cannot be empty")
        self._type_configs.pop(type_path, None)

    def clear_types(self) -> None:
        """Remove all type configurations."""
        self._type_configs.clear()

    # ----- Resolvers / Helpers -----

    def list_types(self) -> list[str]:
        """List configured type names (in insertion order)."""
        return list(self._type_configs.keys())

    def iter_type_configs(self):
        """Iterate over (type_name, TypeConfig) pairs."""
        return self._type_configs.items()

    def get_type_feature(self, type_name: str) -> str | None:
        """Return the configured feature name for a type (or None)."""
        cfg = self._type_configs.get(type_name)
        return cfg.feature if cfg else None

    def resolve_label(self, fs: FeatureStructure, type_name: str) -> str | None:
        """Compute a display label for a feature structure of the given type."""
        cfg = self._type_configs.get(type_name)
        if cfg is None:
            return None
        value = self._get_feature_value(fs, cfg.feature)
        return cfg.label_for_value(value)

    def resolve_color(self, type_name: str, label: str | None) -> str | None:
        """Compute a display color for the given type and label."""
        cfg = self._type_configs.get(type_name)
        if cfg is None:
            return None
        return cfg.color_for_label(label)

    def _get_feature_value(self, fs: FeatureStructure, feature_name: str | None) -> Any:
        """Safely get a feature value (raises VisualizerException on missing feature)."""
        if feature_name is None:
            return None
        try:
            return fs.get(feature_name)
        except KeyError as e:
            raise VisualizerException(
                f"Feature '{feature_name}' not found on type '{fs.type.name}'"
            ) from e

    @staticmethod
    def _wrap_html_page(fragment: str, title: str = "Visualizer") -> str:
        """
        Wrap an HTML fragment into a full document with UTF‑8 meta.

        Useful to avoid browser encoding issues for fragments rendered by spaCy's displaCy.
        """
        return (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n<meta charset=\"utf-8\">\n"
            f"<title>{title}</title>\n</head>\n<body>\n"
            f"{fragment}\n"
            "</body>\n</html>"
        )

    @abc.abstractmethod
    def visualize(
        self,
        cas: Cas,
        *,
        start: int = 0,
        end: int = -1,
        output_format: str = "html",
    ) -> str:
        """
        Build and render a visualization.

        Parameters:
        - cas: CAS instance to visualize
        - start, end: character range to consider (end = -1 means end of document)
        - output_format: format identifier (subclasses validate supported values)

        Returns:
        - a single string (HTML, SVG, CSV, JSON, LaTeX depending on the visualizer and selected format)

        Errors:
        - VisualizerException on unsupported format or when strict mode requires content but none is found.
        """
        raise NotImplementedError
