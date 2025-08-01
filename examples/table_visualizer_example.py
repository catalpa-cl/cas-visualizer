import tempfile
import webbrowser

from cas_visualizer.visualizer2 import TableVisualizer

cas = '../data/hagen.txt.xmi'
ts = '../data/TypeSystem.xml'

span_vis = TableVisualizer()
span_vis.load_cas(cas, ts)

span_vis.add_type(type_path='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity')
span_vis.add_type(type_path='de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly', feature_name='description')

html = span_vis.visualize().reset_index(drop=True).to_html()

### Render HTML in Browser

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='cp1252') as f:
    url = 'file://' + f.name
    f.write(html)
webbrowser.open(url)