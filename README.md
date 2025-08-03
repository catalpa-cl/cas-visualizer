## Overview

The `cas_visualizer`library can be used to transform a *Common Analysis System (CAS)* data structure into an annotated html string.

## Quick start

##### from [self-contained example](https://github.com/catalpa-cl/cas-visualizer/blob/3bd8cb9454010a48e274deb88c4c80b39e1c17e9/examples/spacy_visualization_example.py) :
We require a *CAS file* or `cassis.Cas` object that contains a text, e.g.:

```
Die Fernuniversität in Hagen (Eigenschreibweise: FernUniversität) ist die erste und einzige staatliche Fernuniversität in Deutschland. Ihr Sitz befindet sich in Hagen in Nordrhein-Westfalen. Nach Angaben des Statistischen Bundesamtes war sie, ohne Berücksichtigung von Akademie- und Weiterbildungsstudierenden, mit über 76.000 Studierenden im Wintersemester 2016/2017[3] die größte deutsche Universität.[4]
```

The *CAS* is based on a *typesystem file* or `cassis.TypeSystem` object and specifies annotation types, e.g.: 

`de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity`

To transform this annotation into a highlighted html string, we run:

```
from cas_visualizer.visualizer import SpanVisualiser

cas = '../data/hagen.txt.xmi'
ts = '../data/TypeSystem.xml'

span_vis = SpanVisualizer(ts)

span_vis.add_type(type_name='NamedEntity')

html = span_vis.visualize(cas)
```
Finally, in a browser we can then render it:

![Screenshot_1](https://raw.githubusercontent.com/catalpa-cl/cas-visualizer/5619438ed444b62392f76ac7151ee6c7c9d8ea61/img/readme_img.png)

### How to publish

1) Increase the version number in `pyproject.toml`
2) Run `poetry build`
3) [Optional] If no token is configured:
   * Create an API-Token by visiting: https://pypi.org/manage/account/#api-tokens
   * Replace `TOKEN` with the string of the API-Token and run `poetry config pypi-token.pypi TOKEN`
4) Run `poetry publish`