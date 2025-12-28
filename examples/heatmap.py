import tempfile
import webbrowser
from pathlib import Path
from cas_visualizer.util import ensure_cas, ensure_typesystem
from cas_visualizer.visualizer import HeatmapVisualizer

ts = ensure_typesystem(Path("data/TypeSystem.xml"))
cas = ensure_cas(Path("data/hagen.txt.xmi"), ts)

TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
NER  = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"

vis = HeatmapVisualizer(
    ts,
    token_type=TOKEN,
    types=[NER],          # annotations to visualize
    mode="binary",        # or "density"
    max_cols=40,
    cell_px=3,
    width_px=None,        # let cell_px define display size; or set a target width here
    page=True,
    strict=True,
)

html = vis.visualize(cas, start=0, end=-1, output_format="html")
# Render HTML in Browser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
    f.write(html)
    url = Path(f.name).as_uri()
webbrowser.open(url)
