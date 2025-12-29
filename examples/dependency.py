import tempfile
import webbrowser

from pathlib import Path
from cas_visualizer.util import ensure_cas, ensure_typesystem
from cas_visualizer.visualizer import SpacyDependencyVisualizer, SpacyDependencyVisualizerConfig

# Load TypeSystem
ts = ensure_typesystem(Path(__file__).parent.parent / 'data' / 'dakoda_typesystem.xml')

config = SpacyDependencyVisualizerConfig(
    dep_type='org.dakoda.syntax.UDependency',
    pos_type='de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS',
    span_type='de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence',
    # feature_config/feature_map can be omitted if defaults match
)
# Construct the visualizer (no per-call options; set them here if needed)
dep_vis = SpacyDependencyVisualizer(
    ts,
    config
    # renderer-specific options live in the constructor
    # minify=False, page=False, options=None by default
)

# Load CAS and select a view
cas = ensure_cas(Path(__file__).parent.parent / 'data' / 'SWI03_fD_Mo107_c.xmi', ts).get_view('ctok')

# Visualize: default range (start=0, end=-1) and format (html)
html = dep_vis.visualize(cas)

# Visualize with explicit start
html_start = dep_vis.visualize(cas, start=0)

# Visualize with a limited range; options are not passed here anymore
html_range = dep_vis.visualize(cas, start=0, end=100)

# If you need renderer options (color, compact), create another visualizer with options
dep_vis_blue = SpacyDependencyVisualizer(
    ts,
    config,
    options={'color': 'blue', 'compact': True},
    minify=True,   # optional
    page=False,    # optional
)
html = dep_vis_blue.visualize(cas, start=0, end=100)

# Alternatively: build → render explicitly
spec = dep_vis.build(cas, start=0, end=100)
html_from_spec = dep_vis.render(spec, output_format='html')
svgs_from_spec = dep_vis.render(spec, output_format='svg')

# Render HTML in browser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
    f.write(html)
    url = 'file://' + f.name
webbrowser.open(url)