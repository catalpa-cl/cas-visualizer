from __future__ import annotations

from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

from cassis import Cas, TypeSystem

from cas_visualizer._base import Visualizer, VisualizerException


class HeatmapVisualizer(Visualizer):
    """
    Token-level heat map showing where configured annotations occur, one cell per token.

    Visual semantics:
    - 'binary': a token-cell is filled if any configured annotation overlaps the token (annotated color).
    - 'density': a token-cell opacity scales with the number of overlapping annotations (annotated color).
    - Non-annotated tokens get a contrasting fill color to make sentence length and structure visible.

    Row modes:
    - 'flat': grid wraps after max_cols (single token stream).
    - 'sentence': one row per sentence; each row ends at its sentence boundary (ragged layout).

    Grid geometry:
    - flat: cols = min(token_count, max_cols); rows = ceil(token_count / cols).
    - sentence: cols = max token count across all sentences in range; rows = number of sentences.

    Parameters:
    - ts: TypeSystem | Path | str
    - token_type: fully qualified CAS type name of tokens
      e.g., "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
    - sentence_type: sentence annotation type; defaults to short "Sentence" with DKPro fallback.
    - types: list[str] | None
      CAS annotation types to include in the heat map (use add_type(...) as alternative).
    - mode: 'binary' | 'density' (default: 'binary')
    - row_mode: 'sentence' | 'flat' (default: 'sentence')
    - color_hex: '#RRGGBB' annotated cell base color (default: '#FF4500' orangered)
    - unannotated_hex: '#RRGGBB' non-annotated cell color (default: complement of color_hex)
    - background_hex: '#RRGGBB' background (default: '#FFFFFF')
    - ann_alpha_max: max alpha for annotated cells (default: 0.9)
    - unann_alpha: alpha for non-annotated cells (default: 0.25)
    - max_cols: maximum logical columns for 'flat' mode (default: 40)
    - cell_px: CSS pixels per logical cell (default: 2)
    - width_px: optional CSS target width (default: None). If provided, pixels-per-cell ≈ floor(width_px / cols).
    - page: wrap fragment into a full HTML page (UTF‑8) if True (default: False)
    - strict: raise when no tokens or no coverage are found (default: True)

    Range semantics:
    - Only tokens fully inside [start, end) are part of the grid.
    - An annotation overlaps a token if their half-open intervals intersect:
      token [t_b, t_e) overlaps fs [f_b, f_e) iff t_b < f_e and f_b < t_e.
    """

    def __init__(
        self,
        ts: TypeSystem | Path | str,
        *,
        token_type: str,
        sentence_type: str = "Sentence",
        types: list[str] | None = None,
        mode: str = "binary",
        row_mode: str = "sentence",
        color_hex: str = "#FF4500",
        unannotated_hex: str | None = None,
        background_hex: str = "#FFFFFF",
        ann_alpha_max: float = 0.9,
        unann_alpha: float = 0.25,
        max_cols: int = 40,
        cell_px: int = 2,
        width_px: int | None = None,
        page: bool = False,
        strict: bool = True,
    ):
        # Accept Path by converting to str for base init
        super().__init__(ts if isinstance(ts, TypeSystem) else str(ts))

        if not token_type:
            raise VisualizerException("HeatmapVisualizer: token_type must be provided")
        self._token_type = token_type

        if mode not in ("binary", "density"):
            raise VisualizerException("HeatmapVisualizer: mode must be 'binary' or 'density'")
        self._mode = mode

        if row_mode not in ("sentence", "flat"):
            raise VisualizerException("HeatmapVisualizer: row_mode must be 'sentence' or 'flat'")
        self._row_mode = row_mode

        if max_cols <= 0:
            raise VisualizerException("HeatmapVisualizer: max_cols must be > 0")
        self._max_cols = max_cols

        if cell_px <= 0:
            raise VisualizerException("HeatmapVisualizer: cell_px must be > 0")
        self._cell_px = cell_px

        if width_px is not None and width_px <= 0:
            raise VisualizerException("HeatmapVisualizer: width_px must be > 0 if provided")
        self._width_px = width_px

        self._sentence_type = sentence_type
        self._sentence_fallback = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

        self._color_hex = color_hex
        self._background_hex = background_hex
        # auto-compute a contrasting color if not provided (simple complement)
        self._unannotated_hex = unannotated_hex or self._hex_complement(color_hex)

        if not (0.0 < ann_alpha_max <= 1.0):
            raise VisualizerException("HeatmapVisualizer: ann_alpha_max must be in (0, 1]")
        if not (0.0 < unann_alpha <= 1.0):
            raise VisualizerException("HeatmapVisualizer: unann_alpha must be in (0, 1]")
        self._ann_alpha_max = ann_alpha_max
        self._unann_alpha = unann_alpha

        self._page = page
        self._strict = strict

        if types:
            for t in types:
                self.add_type(t)

    # ---------- build / render / visualize ----------

    def build(self, cas: Cas, *, start: int = 0, end: int = -1) -> Dict[str, Any]:
        text_len = len(cas.sofa_string)
        if end == -1:
            end = text_len
        if start < 0 or end < 0 or start > end or end > text_len:
            raise VisualizerException(
                f"HeatmapVisualizer: invalid range [{start}, {end}] for document length {text_len}"
            )

        type_names = self.list_types()
        if not type_names:
            raise VisualizerException("HeatmapVisualizer: no annotation types configured. Use add_type(...) first.")

        if self._row_mode == "flat":
            tokens = [tok for tok in cas.select(self._token_type) if tok.begin >= start and tok.end <= end]
            if not tokens:
                raise VisualizerException(
                    f"HeatmapVisualizer: no tokens of type {self._token_type} found inside range [{start}, {end}]."
                )

            n = len(tokens)
            begins = [t.begin for t in tokens]
            ends = [t.end for t in tokens]

            diff = [0] * (n + 1)
            have_any_overlap = False

            for type_name in type_names:
                for fs in cas.select(type_name):
                    fb = max(fs.begin, start)
                    fe = min(fs.end, end)
                    if fb >= fe:
                        continue
                    li = bisect_right(ends, fb)
                    ri_excl = bisect_left(begins, fe)
                    if li < ri_excl:
                        diff[li] += 1
                        diff[ri_excl] -= 1
                        have_any_overlap = True

            coverage = [0] * n
            cur = 0
            max_count = 0
            for i in range(n):
                cur += diff[i]
                coverage[i] = cur
                if cur > max_count:
                    max_count = cur

            if self._strict and not have_any_overlap:
                raise VisualizerException(
                    f"HeatmapVisualizer: no overlaps found for types {type_names} over tokens in range [{start}, {end}]."
                )
            if self._strict and max_count == 0:
                raise VisualizerException(
                    f"HeatmapVisualizer: nothing to render (no covered tokens) in range [{start}, {end}]."
                )

            cols = min(n, self._max_cols)
            rows = (n + cols - 1) // cols  # ceil
            css_cell_px, css_width_px = self._compute_css_cell_and_width(cols)
            css_height_px = rows * css_cell_px

            r, g, b = self._hex_to_rgb(self._color_hex)
            ur, ug, ub = self._hex_to_rgb(self._unannotated_hex)
            br, bg, bb = self._hex_to_rgb(self._background_hex)

            return {
                "fill_mode": self._mode,
                "row_mode": "flat",
                "color": [r, g, b],
                "unannotated_color": [ur, ug, ub],
                "background": [br, bg, bb],
                "ann_alpha_max": self._ann_alpha_max,
                "unann_alpha": self._unann_alpha,
                "cols": cols,
                "rows": rows,
                "coverage": coverage,   # 1-D coverage
                "max_count": max_count,
                "css_cell_px": css_cell_px,
                "css_width_px": css_width_px,
                "css_height_px": css_height_px,
            }

        # row_mode == 'sentence'
        sentences = list(cas.select(self._sentence_type))
        if not sentences and "." not in self._sentence_type:
            sentences = list(cas.select(self._sentence_fallback))
        sentences = [s for s in sentences if s.begin >= start and s.end <= end]
        if not sentences:
            raise VisualizerException(
                f"HeatmapVisualizer: no sentences of type {self._sentence_type} found inside range [{start}, {end}]."
            )

        sent_tokens: list[list[FeatureStructure]] = []
        tokens_per_row: list[int] = []
        for s in sentences:
            toks = list(cas.select_covered(self._token_type, covering_annotation=s))
            toks = [t for t in toks if t.begin >= start and t.end <= end]
            sent_tokens.append(toks)
            tokens_per_row.append(len(toks))

        if all(n == 0 for n in tokens_per_row):
            raise VisualizerException(
                f"HeatmapVisualizer: sentences contain no tokens of type {self._token_type} inside range [{start}, {end}]."
            )

        have_any_overlap = False
        max_count = 0
        max_cols_for_sentences = max(tokens_per_row)
        rows = len(sent_tokens)
        cols = max_cols_for_sentences

        coverage_rows: list[list[int]] = []
        for s_idx, s in enumerate(sentences):
            toks = sent_tokens[s_idx]
            n = len(toks)
            begins = [t.begin for t in toks]
            ends = [t.end for t in toks]
            diff = [0] * (n + 1)

            for type_name in type_names:
                for fs in cas.select_covered(type_name, covering_annotation=s):
                    fb = max(fs.begin, start)
                    fe = min(fs.end, end)
                    if fb >= fe:
                        continue
                    li = bisect_right(ends, fb)
                    ri_excl = bisect_left(begins, fe)
                    if li < ri_excl:
                        diff[li] += 1
                        diff[ri_excl] -= 1
                        have_any_overlap = True

            row_cov = [0] * n
            cur = 0
            for i in range(n):
                cur += diff[i]
                row_cov[i] = cur
                if cur > max_count:
                    max_count = cur

            # Pad with zeros to max_cols_for_sentences (padded cells represent "no token")
            if n < max_cols_for_sentences:
                row_cov = row_cov + [0] * (max_cols_for_sentences - n)

            coverage_rows.append(row_cov)

        if self._strict and not have_any_overlap:
            raise VisualizerException(
                f"HeatmapVisualizer: no overlaps found for types {type_names} over sentences in range [{start}, {end}]."
            )
        if self._strict and max_count == 0:
            raise VisualizerException(
                f"HeatmapVisualizer: nothing to render (no covered tokens) in range [{start}, {end}]."
            )

        coverage = [c for row in coverage_rows for c in row]

        css_cell_px, css_width_px = self._compute_css_cell_and_width(cols)
        css_height_px = rows * css_cell_px

        r, g, b = self._hex_to_rgb(self._color_hex)
        ur, ug, ub = self._hex_to_rgb(self._unannotated_hex)
        br, bg, bb = self._hex_to_rgb(self._background_hex)

        return {
            "fill_mode": self._mode,
            "row_mode": "sentence",
            "color": [r, g, b],
            "unannotated_color": [ur, ug, ub],
            "background": [br, bg, bb],
            "ann_alpha_max": self._ann_alpha_max,
            "unann_alpha": self._unann_alpha,
            "cols": cols,
            "rows": rows,
            "coverage": coverage,               # flattened row-major
            "tokens_per_row": tokens_per_row,   # differentiate padded cells vs real tokens
            "max_count": max_count,
            "css_cell_px": css_cell_px,
            "css_width_px": css_width_px,
            "css_height_px": css_height_px,
        }

    def render(self, spec: Dict[str, Any], *, output_format: str = "html") -> str:
        fmt = output_format.lower()
        if fmt != "html":
            raise VisualizerException("HeatmapVisualizer supports only 'html' output_format")
        frag = self._render_canvas_fragment(spec)
        return self._wrap_html_page(frag, "HeatmapVisualizer") if self._page else frag

    def visualize(self, cas: Cas, *, start: int = 0, end: int = -1, output_format: str = "html") -> str:
        spec = self.build(cas, start=start, end=end)
        return self.render(spec, output_format=output_format)

    # ---------- internal helpers ----------

    def _compute_css_cell_and_width(self, cols: int) -> tuple[int, int]:
        """
        Compute CSS pixels per logical cell and total width.
        - If width_px is None: use cell_px exactly.
        - If width_px is provided: respect requested cell_px where possible,
        but reduce it if necessary so that cols * css_cell_px <= width_px.
        """
        if self._width_px is None:
            css_cell_px = self._cell_px
            css_width_px = cols * css_cell_px
        else:
            # Width-imposed cell size (integer pixels per cell)
            imposed = self._width_px // max(cols, 1)
            if imposed < 1:
                imposed = 1
            # Use the smaller of requested and imposed to respect the width constraint
            css_cell_px = min(self._cell_px, imposed)
            css_width_px = cols * css_cell_px
        return css_cell_px, css_width_px

    @staticmethod
    def _hex_to_rgb(hx: str) -> tuple[int, int, int]:
        """
        Parse hex color code to RGB tuple.

        Handles invalid formats by returning default orangered.
        A warning could be logged here if logging is added.

        Parameters:
        - hx: hex color string (e.g., '#FF4500')

        Returns:
        - tuple of (red, green, blue) in range [0-255]
        """
        s = hx.strip().lstrip("#")
        if len(s) != 6:
            # Invalid hex format - return default orangered
            return (255, 69, 0)
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            # Non-hex characters - return default orangered
            return (255, 69, 0)

    @staticmethod
    def _hex_complement(hx: str) -> str:
        s = hx.strip().lstrip("#")
        if len(s) != 6:
            return "#00B2FF"  # some contrasting default (azure)
        try:
            r = 255 - int(s[0:2], 16)
            g = 255 - int(s[2:4], 16)
            b = 255 - int(s[4:6], 16)
            return f"#{r:02X}{g:02X}{b:02X}"
        except ValueError:
            # Non-hex characters - return default contrasting color
            return "#00B2FF"

    def _render_canvas_fragment(self, spec: Dict[str, Any]) -> str:
        import json

        fill_mode = spec["fill_mode"]
        color = spec["color"]
        unann = spec["unannotated_color"]
        bg = spec["background"]
        ann_alpha_max = float(spec["ann_alpha_max"])
        unann_alpha = float(spec["unann_alpha"])
        cols = int(spec["cols"])
        rows = int(spec["rows"])
        coverage: list[int] = spec["coverage"]
        tokens_per_row: list[int] | None = spec.get("tokens_per_row")
        max_count = int(spec["max_count"])
        css_width_px = int(spec["css_width_px"])
        css_height_px = int(spec["css_height_px"])

        data = {
            "fillMode": fill_mode,
            "annColor": color,
            "unannColor": unann,
            "background": bg,
            "annAlphaMax": ann_alpha_max,
            "unannAlpha": unann_alpha,
            "cols": cols,
            "rows": rows,
            "coverage": coverage,
            "tokensPerRow": tokens_per_row,
            "maxCount": max_count,
        }
        js_data = json.dumps(data)

        return f"""
<div style="display:inline-block;border:1px solid #ccc;padding:6px;background:#f9f9f9;">
  <div style="font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial;">
    <strong>Token heat map</strong><br>cols={cols}, rows={rows}, mode={fill_mode}
  </div>
  <canvas id="hm_canvas" width="{cols}" height="{rows}"
          style="width:{css_width_px}px;height:{css_height_px}px;image-rendering:pixelated;border:1px solid #ddd;"></canvas>
</div>
<script>
(function() {{
  const data = {js_data};
  const cvs = document.getElementById('hm_canvas');
  const ctx = cvs.getContext('2d');

  const cols = data.cols;
  const rows = data.rows;
  const cov = data.coverage;
  const maxC = data.maxCount;
  const ann = data.annColor;    // [r,g,b]
  const unann = data.unannColor; // [r,g,b]
  const bg = data.background;
  const annA = data.annAlphaMax;
  const unannA = data.unannAlpha;
  const tpr = data.tokensPerRow; // null in flat mode

  // Fill background
  ctx.fillStyle = 'rgb(' + bg[0] + ',' + bg[1] + ',' + bg[2] + ')';
  ctx.fillRect(0, 0, cvs.width, cvs.height);

  // Draw cells: one logical pixel per token; CSS scales via style width/height
  let k = 0;
  for (let y = 0; y < rows; y++) {{
    const rowLen = tpr ? tpr[y] : cols; // real tokens in this row
    for (let x = 0; x < cols; x++) {{
      if (k >= cov.length) break;
      const c = cov[k];
      const isRealToken = x < rowLen;
      if (isRealToken) {{
        if (data.fillMode === 'binary') {{
          if (c > 0) {{
            ctx.fillStyle = 'rgba(' + ann[0] + ',' + ann[1] + ',' + ann[2] + ',' + annA + ')';
            ctx.fillRect(x, y, 1, 1);
          }} else {{
            ctx.fillStyle = 'rgba(' + unann[0] + ',' + unann[1] + ',' + unann[2] + ',' + unannA + ')';
            ctx.fillRect(x, y, 1, 1);
          }}
        }} else {{
          if (c > 0 && maxC > 0) {{
            const alpha = Math.min(annA, 0.1 + (annA - 0.1) * (c / maxC));
            ctx.fillStyle = 'rgba(' + ann[0] + ',' + ann[1] + ',' + ann[2] + ',' + alpha + ')';
            ctx.fillRect(x, y, 1, 1);
          }} else {{
            ctx.fillStyle = 'rgba(' + unann[0] + ',' + unann[1] + ',' + unann[2] + ',' + unannA + ')';
            ctx.fillRect(x, y, 1, 1);
          }}
        }}
      }}
      k++;
    }}
  }}
}})();
</script>
""".strip()