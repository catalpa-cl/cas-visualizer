## Overview

The `cas_visualizer` library provides multiple ways to visualize Common Analysis System (CAS) annotations from [dkpro-cassis](https://github.com/dkpro-cassis/dkpro-cassis) and [Udapi](https://udapi.github.io/). It supports rendering in various formats:

- **Spacy-style HTML spans** - Interactive span visualizations using [spaCy's displaCy](https://spacy.io/api/displacy)
- **Dependency trees** - Both UDPipe format and spaCy-style HTML
- **Tables** - CSV/HTML tabular representation of annotations
- **Heatmaps** - Matplotlib heatmaps showing annotation density
- **DOCX** - Microsoft Word documents with colored span annotations

## Quick start

*(see [examples](examples/) for complete implementations)*

### 1. Basic Span Visualization

We require a CAS file or `cassis.Cas` object containing text:

```python
from cassis import load_cas_from_xmi, load_typesystem
from cas_visualizer import SpacySpanVisualizer

# Load CAS and TypeSystem
cas = load_cas_from_xmi('../data/hagen.txt.xmi', typesystem=load_typesystem('../data/TypeSystem.xml'))
ts = load_typesystem('../data/TypeSystem.xml')

# Create visualizer
vis = SpacySpanVisualizer(ts)

# Configure annotation types
vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity', color='lightblue')

# Render to HTML
html = vis.visualize(cas)
print(html)  # Display in browser or save to file
```

### 2. Configuration Examples

**Map feature values to labels and colors:**

```python
vis.add_feature(
    name='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity',
    feature='value',
    value='PERSON',
    label='Person',
    color='lightblue'
)
vis.add_feature(
    name='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity',
    feature='value',
    value='LOCATION',
    label='Location',
    color='lightgreen'
)
```

**Highlighting mode (instead of underlines):**

```python
# Default is underline via SpanRenderer
# Use HIGHLIGHT mode via EntityRenderer
vis = SpacySpanVisualizer(ts)
vis.render_mode = "HIGHLIGHT"  # or "UNDERLINE" (default)

html = vis.visualize(cas)
```

### 3. Other Visualizers

**Dependency trees:**

```python
from cas_visualizer import SpacyDependencyVisualizer, UdapiDependencyVisualizer

# spaCy-style HTML
dep_vis = SpacyDependencyVisualizer(ts)
html = dep_vis.visualize(cas)

# UDPipe format (string-based)
udapi_vis = UdapiDependencyVisualizer(ts)
conllu = udapi_vis.visualize(cas)
```

**Tables:**

```python
from cas_visualizer import TableVisualizer

table_vis = TableVisualizer(ts)
table_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity')

# CSV format
csv_output = table_vis.visualize(cas, output_format='csv')

# HTML format
html_output = table_vis.visualize(cas, output_format='html')
```

**Heatmaps:**

```python
from cas_visualizer import HeatmapVisualizer

heatmap_vis = HeatmapVisualizer(ts)
heatmap_vis.add_type(name='de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity')

# Returns matplotlib Figure object
fig = heatmap_vis.visualize(cas)
fig.show()  # or fig.savefig('heatmap.png')
```

---

## API Reference

### Core Classes

All visualizers inherit from the `Visualizer` base class:

```python
class Visualizer(abc.ABC):
    """Base class for CAS visualizers."""
    
    def __init__(self, ts: str | Path | TypeSystem):
        """Initialize with TypeSystem (file path or TypeSystem object)."""
    
    def add_type(self, name: str, feature: str | None = None, 
                 color: str | None = None, label: str | None = None) -> None:
        """Register a CAS type for visualization."""
    
    def add_feature(self, name: str, feature: str, value: Any,
                    color: str | None = None, label: str | None = None) -> None:
        """Map specific feature values to labels and colors."""
    
    def visualize(self, cas: Cas, *, start: int = 0, end: int = -1, 
                  output_format: str = "html") -> str:
        """Build and render visualization."""
    
    def list_types(self) -> list[str]:
        """List registered type names."""
    
    def clear_types(self) -> None:
        """Clear all type configurations."""
```

### Available Visualizers

- **`SpacySpanVisualizer`** - HTML span visualization (underline or highlight)
- **`DocxSpanVisualizer`** - DOCX document with colored spans  
- **`TableVisualizer`** - Tabular representation (CSV/HTML)
- **`SpacyDependencyVisualizer`** - spaCy-style dependency tree HTML
- **`UdapiDependencyVisualizer`** - UDPipe conllu format
- **`HeatmapVisualizer`** - Matplotlib annotation density heatmap

---

## Architecture

The library is organized into separate modules for each visualizer type:

- `cas_visualizer/_base.py` - Base classes (`Visualizer`, `VisualizerException`, `TypeConfig`)
- `cas_visualizer/span.py` - Span visualizers
- `cas_visualizer/dependency.py` - Dependency tree visualizers
- `cas_visualizer/table.py` - Table visualizer
- `cas_visualizer/heatmap.py` - Heatmap visualizer
- `cas_visualizer/util.py` - Utility functions
- `cas_visualizer/__init__.py` - Public API exports

All visualizers follow a consistent interface:
1. **`build(cas, start, end)`** - Build internal representation
2. **`render(spec, output_format)`** - Render to output format
3. **`visualize(cas, output_format)`** - Convenience method combining both

---

## Development

### Setup

```bash
git clone https://github.com/zesch/cas-visualizer.git
cd cas-visualizer
poetry install
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=cas_visualizer

# Specific test file
poetry run pytest tests/test_span_visualizer.py -v
```

### Code Quality

```bash
# Format code
poetry run black cas_visualizer/ tests/
poetry run isort cas_visualizer/ tests/

# Lint
poetry run flake8 cas_visualizer/ tests/

# Type check
poetry run mypy cas_visualizer/

# Or use pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### CI/CD

The project uses GitHub Actions for:
- **Testing** on Python 3.11, 3.12, 3.13 (Linux, macOS, Windows)
- **Linting** with Black, isort, flake8
- **Type checking** with mypy
- **Coverage** reporting

Workflows run on every push and pull request to `main` and `develop` branches.

---

## How to Publish

Only for maintainers:

1. Update version in `pyproject.toml`
2. Run `poetry build`
3. Push to GitHub - CI/CD will handle the rest (when release automation is configured)

Or manually:
```bash
poetry publish --repository pypi
```