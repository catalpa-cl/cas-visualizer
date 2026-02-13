import json
import csv
from io import StringIO

import pandas as pd
import pytest
from cassis import Cas

from cas_visualizer.visualizer import TableVisualizer, VisualizerException

from tests.fixtures import *

TS = 'data/TypeSystem.xml'
T_ENT = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"


@pytest.fixture()
def cas_entities(typesystem):
    """
    CAS with text "I saw a dog."
    Entities:
      - "saw" value="VERB"
      - "dog" value="ANIMAL <strong>bold</strong>" (to test HTML escaping)
    """
    cas = Cas(typesystem=typesystem)
    cas.sofa_string = "I saw a dog."

    ENT_T = typesystem.get_type(T_ENT)

    e_saw = ENT_T(begin=2, end=5, value="VERB")
    e_dog = ENT_T(begin=8, end=11, value="ANIMAL <strong>bold</strong>")
    cas.add(e_saw)
    cas.add(e_dog)

    return cas


def test_build_basic_dataframe(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")

    df = tv.build(cas_entities)

    # Expected columns and row count
    assert list(df.columns) == ["text", "feature", "value", "begin", "end"]
    assert len(df) == 2

    # Sorted by begin/end by default
    texts = list(df["text"])
    assert texts == ["saw", "dog"]

    # Feature/value mapping as configured
    assert set(df["feature"]) == {"value"}
    assert set(df["value"]) == {"VERB", "ANIMAL <strong>bold</strong>"}


def test_build_empty_when_no_types(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    df = tv.build(cas_entities)
    assert df.empty


def test_visualize_equals_render_default_html(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)  # default_format='html'
    tv.add_type(T_ENT, feature="value")

    # visualize should equal render(build(...)) for default format
    expected = tv.render(tv.build(cas_entities), output_format="html")
    actual = tv.visualize(cas_entities)
    assert isinstance(actual, str)
    assert actual == expected


def test_render_html(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")

    df = tv.build(cas_entities)
    html = tv.render(df, output_format="html")
    assert isinstance(html, str)
    assert "<table" in html
    assert "saw" in html and "dog" in html


def test_render_csv(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")

    df = tv.build(cas_entities)
    csv_text = tv.render(df, output_format="csv")

    # Parse CSV to verify headers and rows
    sio = StringIO(csv_text)
    reader = csv.reader(sio)
    rows = list(reader)
    # First row is header
    assert rows[0] == ["text", "feature", "value", "begin", "end"]
    # Next rows contain our values
    assert rows[1][0] == "saw"
    assert rows[2][0] == "dog"


def test_render_json_default_records(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")

    df = tv.build(cas_entities)
    json_text = tv.render(df, output_format="json")
    data = json.loads(json_text)
    assert isinstance(data, list)
    assert len(data) == 2
    assert set(data[0].keys()) == {"text", "feature", "value", "begin", "end"}
    assert {d["text"] for d in data} == {"saw", "dog"}


def test_render_json_orient_split(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")

    df = tv.build(cas_entities)
    json_text = tv.render(df, output_format="json", render_options={"orient": "split"})
    data = json.loads(json_text)
    # Expect 'columns' and 'data' keys for split orient
    assert set(data.keys()) == {"index", "columns", "data"}
    assert data["columns"] == ["text", "feature", "value", "begin", "end"]
    assert len(data["data"]) == 2


def test_render_latex(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")

    df = tv.build(cas_entities)
    tex = tv.render(df, output_format="latex")
    assert isinstance(tex, str)
    assert "\\begin{tabular" in tex
    assert "saw" in tex and "dog" in tex


def test_render_unsupported_format(typesystem, cas_entities):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")

    df = tv.build(cas_entities)
    with pytest.raises(VisualizerException):
        _ = tv.render(df, output_format="xlsx")  # unsupported


# ------------------------------
# TableVisualizer (basic)
# ------------------------------

def test_table_visualizer_visualize_uses_output_format(typesystem, cas_single_sentence):
    tv = TableVisualizer(typesystem)
    tv.add_type(T_ENT, feature="value")
    out = tv.visualize(cas_single_sentence, output_format='json')
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 2
    # content sanity
    texts = {row["text"] for row in data}
    assert texts == {"saw", "dog"}