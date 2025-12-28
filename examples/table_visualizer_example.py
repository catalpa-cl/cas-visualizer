import tempfile
import webbrowser
from pathlib import Path
from cas_visualizer.util import ensure_cas, ensure_typesystem
from cas_visualizer.visualizer import TableVisualizer

ts = ensure_typesystem(Path(__file__).parent.parent / 'data' / 'TypeSystem.xml')

table_vis = TableVisualizer(ts)
table_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity', feature='value')
table_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly', feature='description')

cas = ensure_cas(Path(__file__).parent.parent / 'data' / 'hagen.txt.xmi', ts)

# Build DataFrame
df = table_vis.build(cas)
df = df.reset_index(drop=True)

# Export to HTML via the visualizer (respects default_render_options) or via pandas
html = table_vis.render(df, output_format='html')
# alternatively: html = df.to_html(index=False, escape=True)

# Render HTML in Browser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
    f.write(html)
    url = 'file://' + f.name
webbrowser.open(url)