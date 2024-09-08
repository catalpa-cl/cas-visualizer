from cassis import Cas
import spacy


def parse_ents(cas: Cas, labels_to_types: dict, labels_to_colors: dict): # see parse_ents spaCy/spacy/displacy/__init__.py
    tmp_ents = []
    for label in labels_to_types.keys():
        for ent in cas.select(labels_to_types[label]):
            tmp_ents.append(
                {
                    "start": ent.begin,
                    "end": ent.end,
                    "label": label.upper()
                }
            )
    tmp_ents.sort(key=lambda x: (x['start'], x['end']))
    has_overlap = check_overlap(tmp_ents)
    return (spacy.displacy.EntityRenderer({"colors": labels_to_colors}).render_ents(cas.sofa_string, tmp_ents, ""),
            has_overlap)


# requires a sorted list of "tmp_ents" as returned by tmp_ents.sort(key=lambda x: (x['start'], x['end']))
def check_overlap(l_ents):
    for i in range(len(l_ents)):
        start_i = l_ents[i]['start']
        for j in range(len(l_ents)):
            if i != j:
                start_j = l_ents[j]['start']
                end_j = l_ents[j]['end']
                if start_i > end_j:
                    continue
                if start_i == start_j:
                    return True
                if start_i == end_j:
                    return True
                if start_j < start_i < end_j:
                    return True
    return False


def create_tokens(cas_sofa_string: str, feature_structures: []) -> []:
    cas_sofa_tokens = []
    cutting_points = set(_.begin for _ in feature_structures).union(_.end for _ in feature_structures)
    char_index_after_whitespace = set([i + 1 for i, char in enumerate(cas_sofa_string) if char.isspace()])
    cutting_points = cutting_points.union(char_index_after_whitespace)
    prev_point = point = 0
    for point in sorted(cutting_points):
        if point != 0:
            tmp_token = {"start": prev_point, "end": point, "text": cas_sofa_string[prev_point:point]}
            cas_sofa_tokens.append(tmp_token)
            prev_point = point
    if point < len(cas_sofa_string):
        tmp_token = {"start": prev_point, "end": len(cas_sofa_string), "text": cas_sofa_string[prev_point:]}
        cas_sofa_tokens.append(tmp_token)
    return cas_sofa_tokens


def create_spans(cas_sofa_tokens: [], feature_structures: [], label):
    tmp_spans = []
    for fs in feature_structures:
        start_token = 0
        end_token = len(cas_sofa_tokens)
        for idx, token in enumerate(cas_sofa_tokens):
            if token["start"] == fs.begin:
                start_token = idx
            if token["end"] == fs.end:
                end_token = idx + 1

        tmp_spans.append(
            {
                "start": fs.begin,
                "end": fs.end,
                "start_token": start_token,
                "end_token": end_token,
                "label": label.upper()
            }
        )
    return tmp_spans


def parse_spans(cas: Cas, labels_to_types: dict, labels_to_colors: dict) -> str: # see parse_ents spaCy/spacy/displacy/__init__.py
    selected_annotations = [item for typeclass in labels_to_types.values() for item in cas.select(typeclass)]
    tmp_tokens = create_tokens(cas.sofa_string, selected_annotations)
    tmp_token_texts = [_["text"] for _ in sorted(tmp_tokens, key=lambda t: t["start"])]

    tmp_spans = []
    for label in labels_to_types.keys():
        tmp_spans.extend(create_spans(tmp_tokens, cas.select(labels_to_types[label]), label))
    tmp_spans.sort(key=lambda x: x["start"])
    return spacy.displacy.SpanRenderer({"colors": labels_to_colors}).render_spans(tmp_token_texts, tmp_spans, "")
