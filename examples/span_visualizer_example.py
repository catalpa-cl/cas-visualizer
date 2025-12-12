import tempfile
import webbrowser

from cas_visualizer.util import cas_from_string, load_typesystem
from cas_visualizer.visualizer import SpanVisualizer

ts = load_typesystem('../data/TypeSystem.xml')

span_vis = SpanVisualizer(ts)

span_vis.add_type(name='NamedEntity')
#span_vis.add_type(name='NamedEntity', feature="value")
span_vis.add_feature(name='NamedEntity', feature="value", value="MISC", color="rosybrown")
span_vis.add_feature(name='NamedEntity', feature="value", value="LOCATION")
#span_vis.add_feature(name='NamedEntity', feature="identifier", value="XXX")

### uncomment to add another annotation
#span_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly', feature='description')

### uncomment to change span style to highlight
#span_vis.selected_span_type = "HIGHLIGHT"

cas = cas_from_string('../data/hagen.txt.xmi', ts)
html = span_vis.visualize(cas)

### render HTML in Browser

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='cp1252') as f:
    url = 'file://' + f.name
    f.write(html)
webbrowser.open(url)