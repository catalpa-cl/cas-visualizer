import base64
from pathlib import Path
from cas_visualizer.util import ensure_cas, ensure_typesystem
from cas_visualizer.visualizer import DocxSpanVisualizer

ts = ensure_typesystem(Path("data/TypeSystem.xml"))
T_ENT = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"

docx_vis = DocxSpanVisualizer(
    ts,
    mode=DocxSpanVisualizer.HIGHLIGHT,
    types=[T_ENT],
    allow_highlight_overlap=False,
    strict=True
)
# Configure labels/colors (optional)
# docx_vis.add_type(T_ENT, feature="value")
# docx_vis.add_feature(T_ENT, feature="value", value="LOCATION", color="yellow")
cas = ensure_cas(Path("data/hagen.txt.xmi"), ts)

docx_b64 = docx_vis.visualize(cas, start=0, end=500, output_format="docx")
Path("spans.docx").write_bytes(base64.b64decode(docx_b64))
