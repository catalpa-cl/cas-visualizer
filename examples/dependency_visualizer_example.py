import tempfile
import webbrowser

from cas_visualizer.util import cas_from_string, load_typesystem
from cas_visualizer.visualizer import DependencyVisualizer

ts = load_typesystem('../data/dakoda_typesystem.xml')

dep_vis = DependencyVisualizer(ts)

cas = cas_from_string('../data/SWI03_fD_Mo107_c.xmi', ts).get_view('ctok')
html = dep_vis.visualize(cas)
html = dep_vis.visualize(cas, start=0)
html = dep_vis.visualize(cas, start= 0, end=100, options = {'color':'blue', 'compact':True})

svg = dep_vis.visualize(cas, start= 0, end=-1, output_format='svg')[0]

### render HTML in Browser

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='cp1252') as f:
    url = 'file://' + f.name
    f.write(html)
webbrowser.open(url)