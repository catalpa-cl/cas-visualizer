import tempfile
import webbrowser

from cas_visualizer.visualizer import DependencyVisualizer

#cas = '../data/SWI01_fD_Es145_p.xmi'
cas = '../data/SWI03_fD_Mo107_c.xmi'
ts = '../data/dakoda_typesystem.xml'

dep_vis = DependencyVisualizer(ts)

html = dep_vis.visualize(cas, view_name='ctok')
html = dep_vis.visualize(cas, view_name='ctok', span_range=(0,100))
html = dep_vis.visualize(cas, view_name='ctok', span_range=(0,100), options = {'color':'blue', 'compact':True})

### render HTML in Browser

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='cp1252') as f:
    url = 'file://' + f.name
    f.write(html)
webbrowser.open(url)