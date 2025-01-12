from cas_visualizer.visualizer import SpacySpanVisualiser
import cas_visualizer.util as util
import streamlit as st


cas = 'data/hagen.txt.xmi'
ts = 'data/TypeSystem.xml'

spacy_span_vis = SpacySpanVisualiser(util.load_cas(cas, ts), [])

spacy_span_vis.set_selected_annotations_to_types({
    'NAMED_ENTITY': 'de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity',
})
spacy_span_vis.set_annotations_to_colors({
    'NAMED_ENTITY': 'lightgreen',
})
spacy_span_vis.set_span_type(SpacySpanVisualiser.SPAN_STYLE_HIGHLIGHTING)

st.write(spacy_span_vis.visualise(), unsafe_allow_html=True)