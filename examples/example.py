from cassis import load_typesystem, load_cas_from_xmi
import streamlit as st
import pathlib
import sys

from visualizer import VisualisationConfig, TableVisualiser, SpanVisualiser, SpacySpanVisualiser, VisualiserException
import api
import util as util
import cassis.typesystem as types

p = pathlib.Path(__file__).absolute()/ '..' /'src'
sys.path.extend(str(p))

cas = 'data/hagen.txt.xmi'
ts = 'data/TypeSystem.xml'

cfg = VisualisationConfig.from_string('de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS/PosValue')
span_vis = SpanVisualiser(util.load_cas(cas, ts), [cfg])
legend, html = span_vis.visualise()
st.write(legend, unsafe_allow_html=True)
st.write(html, unsafe_allow_html=True)

table_vis = TableVisualiser(util.load_cas(cas, ts), [cfg])
st.table(table_vis.visualise())

spacy_span_vis = SpacySpanVisualiser(util.load_cas(cas, ts), [])
spacy_span_vis.set_selected_annotations_to_types({'POS_NOUN': 'de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS_NOUN'
                                                  ,'POS_NOUN2': 'de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS_NOUN'
                                                  })
spacy_span_vis.set_annotations_to_colors({'POS_NOUN': 'green'})
spacy_span_vis.set_span_type(SpacySpanVisualiser.SPAN_STYLE_HIGHLIGHTING)

try:
    st.write(spacy_span_vis.visualise(), unsafe_allow_html=True)
except VisualiserException as ve:
    st.error(ve)
