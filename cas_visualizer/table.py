from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from cassis import Cas, TypeSystem

from cas_visualizer._base import Visualizer, VisualizerException



# ---------- Table visualizer ----------

class TableVisualizer(Visualizer):
    """
    Tabular visualization of selected annotations.

    Pipeline:
    - build: produce a pandas DataFrame of rows within [start, end)
    - render: export the DataFrame as HTML, CSV, JSON, or LaTeX (returns str)
    - visualize: build + render

    Parameters:
    - ts: TypeSystem or path to TypeSystem file
    - default_render_options: dict of rendering options passed to render()
    - default_sort: whether to sort the DataFrame by (begin, end) by default
    - page: wrap fragment in full HTML page if True
    - strict: raise when DataFrame is empty if True

    Strict mode:
    - If strict=True and the DataFrame is empty, render() raises VisualizerException.
      Otherwise, you may get a valid but empty HTML table/CSV/JSON/LaTeX.
    """
    def __init__(
        self,
        ts: str | Path | TypeSystem,
        *,
        default_render_options: dict[str, Any] | None = None,
        default_sort: bool = True,
        page: bool = False,
        strict: bool = True,
    ):
        super().__init__(ts)
        self._default_render_options = default_render_options or {}
        self._default_sort = default_sort
        self._page = page
        self._strict = strict

    def build(self, cas: Cas, *, start: int = 0, end: int = -1) -> pd.DataFrame:
        """
        Build the table spec (DataFrame). Include only annotations fully inside [start, end).
        If end == -1, treat it as len(document_text).
        """
        if end == -1:
            end = len(cas.sofa_string)
        cols = ["text", "feature", "value", "begin", "end"]
        records: list[dict[str, Any]] = []

        for type_name, cfg in self.iter_type_configs():
            for fs in cas.select(type_name):
                if fs.begin < start or fs.end > end:
                    continue

                feature_name = cfg.feature
                feature_value = self._get_feature_value(fs, feature_name)

                records.append({
                    "text": fs.get_covered_text(),
                    "feature": feature_name,
                    "value": feature_value,
                    "begin": fs.begin,
                    "end": fs.end,
                })

        df = pd.DataFrame.from_records(records, columns=cols)
        if self._default_sort and not df.empty:
            df = df.sort_values(by=["begin", "end"], kind="mergesort")
        return df

    def render(
        self,
        spec: pd.DataFrame,
        *,
        output_format: str = "html",
        render_options: Dict[str, Any] | None = None,
    ) -> str:
        """
        Export the table to the requested format.

        Supported formats:
        - 'html': returns an HTML fragment; wrap to a full page if page=True
        - 'csv': returns CSV text (no index)
        - 'json': returns JSON (records orientation)
        - 'latex': returns LaTeX tabular code

        Errors:
        - VisualizerException on unsupported format or empty result in strict mode.
        """
        fmt = output_format.lower()
        opts = {**self._default_render_options, **(render_options or {})}

        if spec.empty and self._strict:
            raise VisualizerException("TableVisualizer: empty result (no rows in selected range).")

        if fmt == "html":
            frag = spec.to_html(**({"index": False, "escape": True} | opts))
            return self._wrap_html_page(frag, "TableVisualizer") if self._page else frag

        if fmt == "csv":
            return spec.to_csv(**({"index": False} | opts))

        if fmt == "json":
            return spec.to_json(**({"orient": "records"} | opts))

        if fmt == "latex":
            return spec.to_latex(**({"index": False, "escape": True} | opts))

        raise VisualizerException(f"Unsupported table output format: {fmt}")

    def visualize(
        self,
        cas: Cas,
        *,
        start: int = 0,
        end: int = -1,
        output_format: str = "html",
    ) -> str:
        """Convenience wrapper: build + render."""
        spec = self.build(cas, start=start, end=end)
        return self.render(spec, output_format=output_format)
