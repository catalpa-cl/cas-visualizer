import tempfile
import webbrowser

from cas_visualizer.util import cas_from_string, load_typesystem
from cas_visualizer.visualizer import TableVisualizer

ts = load_typesystem('../data/TypeSystem.xml')

table_vis = TableVisualizer(ts)

table_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity')
table_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly', feature='description')


cas = cas_from_string('../data/hagen.txt.xmi', ts)
html = table_vis.visualize(cas).reset_index(drop=True).to_html()

### render HTML in Browser

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='cp1252') as f:
    url = 'file://' + f.name
    f.write(html)
webbrowser.open(url)