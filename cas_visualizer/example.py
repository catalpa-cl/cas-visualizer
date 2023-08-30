from cassis import load_typesystem, load_cas_from_xmi
import streamlit as st
import pathlib
import sys

from visualizer import VisualisationConfig, TableVisualiser
import api
import util as util
import cassis.typesystem as types

p = pathlib.Path(__file__).absolute()/ '..' /'src'
sys.path.extend(str(p))

cas = 'data/hagen.txt.xmi'
ts = 'data/TypeSystem.xml'

cfg = VisualisationConfig.from_string('de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS/PosValue')
vis = TableVisualiser(util.load_cas(cas, ts), [cfg])
vis.visualise()