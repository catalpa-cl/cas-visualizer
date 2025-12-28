import tempfile
import webbrowser
from pathlib import Path
from cas_visualizer.util import ensure_cas, ensure_typesystem
from cas_visualizer.visualizer import SpacySpanVisualizer

ts = ensure_typesystem(Path(__file__).parent.parent / 'data' / 'TypeSystem.xml')

# DKPro full type name for NamedEntity
T_ENT = 'de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity'

span_vis = SpacySpanVisualizer(ts, page=True)

# Configure the type; you can set feature here or via add_feature below
span_vis.add_type(name=T_ENT, feature="value")
# Alternatively: span_vis.add_type(name=T_ENT) and let add_feature assign the feature

# Map specific feature values to colors (LOCATION gets a default color from the iterator)
span_vis.add_feature(name=T_ENT, feature="value", value="MISC", color="rosybrown")
span_vis.add_feature(name=T_ENT, feature="value", value="LOCATION")
# Example of mapping a different feature (uncomment if present in your TS)
# span_vis.add_feature(name=T_ENT, feature="identifier", value="XXX")

# Add another annotation type (optional)
# span_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly', feature='description')

# Change span style to highlight (optional; default is UNDERLINE)
# span_vis.selected_span_type = SpacySpanVisualizer.HIGHLIGHT

cas = ensure_cas(Path(__file__).parent.parent / 'data' / 'hagen.txt.xmi', ts)

# Visualize (output_format must be 'html')
html = span_vis.visualize(cas, output_format='html')

# Render HTML in Browser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
    f.write(html)
    url = Path(f.name).as_uri()
webbrowser.open(url)