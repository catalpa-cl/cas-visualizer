## Overview

The `cas_visualizer`library can be used to transform a Common Analysis System (CAS) data structure into an annotated HTML string. 

The library visualizes annotations with an underlined (default) or highlighted style.

## Quick start

*(see [self-contained example](https://github.com/catalpa-cl/cas-visualizer/blob/2185cff13e35fc60086ee44640768d28f754146b/examples/span_visualizer_example.py))*

We require a CAS file or `cassis.Cas` object that contains a text, e.g.

```
Die Fernuniversität in Hagen (Eigenschreibweise: FernUniversität) ist die erste und einzige staatliche Fernuniversität in Deutschland. Ihr Sitz befindet sich in Hagen in Nordrhein-Westfalen. 
Nach Angaben des Statistischen Bundesamtes war sie, ohne Berücksichtigung von Akademie- und Weiterbildungsstudierenden, mit über 76.000 Studierenden im Wintersemester 2016/2017[3] die größte deutsche Universität.[4]
```

The CAS is based on a type system file or `cassis.TypeSystem` object and specifies annotation types, e.g. 

`de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity`

To transform the text into an HTML string with underlined annotations of type NamedEntity, we run:

```
from cas_visualizer.visualizer import SpanVisualizer

cas = '../data/hagen.txt.xmi'
ts = '../data/TypeSystem.xml'

span_vis = SpanVisualizer(ts)

span_vis.add_type(name='NamedEntity')

html = span_vis.visualize(cas)
```
Finally, in a browser we can then render the HTML string:

![Screenshot_1](https://github.com/catalpa-cl/cas-visualizer/blob/4fb14d0961cc42536a97ab09f6012d5539175f1d/img/readme_img.png?raw=true)

Before visualizing the CAS, you can switch to the highlighted style by calling:

`span_vis.selected_span_type = "HIGHLIGHT"`

![Screenshot_2](https://github.com/catalpa-cl/cas-visualizer/blob/2185cff13e35fc60086ee44640768d28f754146b/img/readme_img2.png?raw=true)

---

### How to publish a new version:

1) Increase the version number in `pyproject.toml`
2) Run `poetry build`
3) [Optional] If no token is configured:
   * Create an API-Token by visiting: https://pypi.org/manage/account/#api-tokens
   * Replace `TOKEN` with the string of the API-Token and run `poetry config pypi-token.pypi TOKEN`
4) Run `poetry publish`