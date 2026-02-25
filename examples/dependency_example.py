import tempfile
import webbrowser

from pathlib import Path
from cas_visualizer.util import ensure_cas, ensure_typesystem
from cas_visualizer.visualizer import (
    SpacyDependencyVisualizer,
    UdapiDependencyVisualizer,
    DependencyVisualizerConfig,
    UdapiTreeConfig,
)

# Load TypeSystem
ts = ensure_typesystem(Path(__file__).parent.parent / 'data' / 'dakoda_typesystem.xml')

config = DependencyVisualizerConfig(
    dep_type='org.dakoda.syntax.UDependency',
    pos_type='de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS',
    span_type='de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence',
)

# Load CAS and select a view
cas = ensure_cas(Path(__file__).parent.parent / 'data' / 'SWI03_fD_Mo107_c.xmi', ts).get_view('ctok')

# ============================================
# Example 1: Spacy-based (HTML/SVG rendering)
# ============================================

dep_vis = SpacyDependencyVisualizer(ts, config)

# Visualize with default settings (HTML output)
html = dep_vis.visualize(cas)

# Visualize with explicit range
html_range = dep_vis.visualize(cas, start=0, end=100)

# Or with explicit output format
svg = dep_vis.visualize(cas, start=0, end=100, output_format='svg')

# Build and render separately for more control
spec = dep_vis.build(cas, start=0, end=100)
html_from_spec = dep_vis.render(spec, output_format='html')

# Render HTML in browser
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
    f.write(html)
    url = 'file://' + f.name
webbrowser.open(url)

# ============================================
# Example 2: UDAPI-based (tree visualization)
# ============================================

# Configure tree display options with optional colorization
tree_config = UdapiTreeConfig(
    attributes="ord,form,lemma,upos,deprel",
    layout="align",
    indent=2,
    mark=None,  # Optional: regex pattern to highlight nodes
    # Optional: colorize specific attribute values
    colorize={
        "misc": {
            "Yes": ["ADV", "SEP", "SVO", "INV", "VEND"]
        },
        "upos": {
            "VERB": ["red", "bold"]
        },
    }
)

udapi_vis = UdapiDependencyVisualizer(ts, config, tree_config=tree_config)

# Visualize as formatted text trees
text_output = udapi_vis.visualize(cas, start=0, end=100)
print(text_output)

# Or build and render separately
trees = udapi_vis.build(cas, start=0, end=100)
tree_text = udapi_vis.render(trees, output_format='text')
print(tree_text)

# ============================================
# Example 3: Advanced colorization
# ============================================

# Configure with more complex colorization patterns and all available options
advanced_tree_config = UdapiTreeConfig(
    attributes="ord,form,lemma,upos,feats,deprel,misc",
    layout="align",
    indent=2,
    mark="(ADV|SVO|TopoField)",  # Highlight nodes matching this regex
    # Colorize multiple attributes with different values and color codes
    colorize={
        # Colorize 'misc' attribute - when value is "Yes", show with these colors
        "misc": {
            "Yes": ["ADV", "SEP", "SVO", "INV", "VEND"]
        },
        # Colorize 'upos' attribute - when POS tag is VERB, show in red bold
        "upos": {
            "VERB": ["red", "bold"],
            "NOUN": ["cyan"],
            "AUX": ["yellow", "bold"],
        },
        # Colorize 'deprel' attribute - highlight certain dependency relations
        "deprel": {
            "nsubj": ["green"],
            "obj": ["magenta"],
            "root": ["white", "on_red"],
        }
    }
)

advanced_udapi_vis = UdapiDependencyVisualizer(ts, config, tree_config=advanced_tree_config)

# Render with all the colorization
colored_output = advanced_udapi_vis.visualize(cas, start=0, end=100)
print("\n=== Colorized Tree Visualization ===")
print(colored_output)

# ============================================
# Example 4: Minimal colorization
# ============================================

# Simple colorization with just one attribute
minimal_tree_config = UdapiTreeConfig(
    attributes="ord,form,upos,deprel",
    layout="compact",
    indent=1,
    colorize={
        "upos": {
            "VERB": ["red", "bold"]
        }
    }
)

minimal_vis = UdapiDependencyVisualizer(ts, config, tree_config=minimal_tree_config)
minimal_output = minimal_vis.visualize(cas)  # Use full document range
print("\n=== Minimal Colorized Output ===")
print(minimal_output)

