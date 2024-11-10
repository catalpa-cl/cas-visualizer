## Overview

The `cas_visualizer`library can be used to transform a `cassis.Cas` object into an annotated html string.

## Quick start

We start out with a `cassis.Cas` object that contains the following text:

```
Die Fernuniversität in Hagen (Eigenschreibweise: FernUniversität) ist die erste und einzige staatliche Fernuniversität in Deutschland. Ihr Sitz befindet sich in Hagen in Nordrhein-Westfalen. Nach Angaben des Statistischen Bundesamtes war sie, ohne Berücksichtigung von Akademie- und Weiterbildungsstudierenden, mit über 76.000 Studierenden im Wintersemester 2016/2017[3] die größte deutsche Universität.[4]
```

and is annotated with a `cassis.TypeSystem` annotation: 

`de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity`

Let's transform this annotation into a `spacy`-styled highlighted html string:

```
from visualizer import SpacySpanVisualiser

spacy_span_vis = SpacySpanVisualiser(cas, [])
annotation = 'de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity'

spacy_span_vis.set_selected_annotations_to_types({'NAMED_ENTITY': annotation)
spacy_span_vis.set_annotations_to_colors({'NAMED_ENTITY': 'lightgreen'})
spacy_span_vis.set_span_type(SpacySpanVisualiser.SPAN_STYLE_HIGHLIGHTING)

html = spacy_span_vis.visualise()
```
Using `streamlit` we can then render it. 

E.g. `st.write(html, unsafe_allow_html=True)`: 

![Screenshot_1](https://raw.githubusercontent.com/catalpa-cl/cas-visualizer/5619438ed444b62392f76ac7151ee6c7c9d8ea61/img/readme_img.png)

### How to publish

1) Increase the version number in `pyproject.toml`
2) Run `poetry build`
3) [Optional] If no token is configured:
   * Create an API-Token by visiting: https://pypi.org/manage/account/#api-tokens
   * Replace `TOKEN` with the string of the API-Token and run `poetry config pypi-token.pypi TOKEN`
4) Run `poetry publish`