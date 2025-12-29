import tempfile
import webbrowser
from pathlib import Path
from cas_visualizer.util import ensure_cas, ensure_typesystem
from cas_visualizer.visualizer import HeatmapVisualizer

ts = ensure_typesystem(Path("data/TypeSystem.xml"))
cas = ensure_cas(Path("data/hagen.txt.xmi"), ts)

TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
NER  = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"

hm = HeatmapVisualizer(
    ts,
    token_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
    sentence_type="Sentence",   # short name, falls back to DKPro full path if needed
    types=["de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"],
    row_mode="sentence",
    cell_px=10,
    mode="binary",
    color_hex="#FF4500",         # annotated cells
    unannotated_hex=None,        # auto-computed contrast
    background_hex="#FFFFFF",
)
hm.add_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity")
html = hm.visualize(cas, output_format="html")

# Render HTML in Browser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
    f.write(html)
    url = Path(f.name).as_uri()
webbrowser.open(url)

hm2 = HeatmapVisualizer(
    ts,
    token_type="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
    row_mode="flat",
    cell_px=5,
    mode="density",
    max_cols=50,
)
hm2.add_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity")
html2 = hm2.visualize(cas, output_format="html")

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
    f.write(html2)
    url = Path(f.name).as_uri()
webbrowser.open(url)