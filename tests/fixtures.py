import pytest
from pathlib import Path
from cassis import Cas
from cas_visualizer.util import ensure_typesystem

@pytest.fixture(scope='module')
def typesystem():
    return ensure_typesystem(Path(__file__).parent.parent / 'data' / 'dakoda_typesystem.xml')

@pytest.fixture()
def cas_single_sentence(typesystem):
    """
    Text: "I saw a dog."
    POS (begin,end,PosValue):
      I: (0,1,PRON)
      saw: (2,5,VERB)
      a: (6,7,DET)
      dog: (8,11,NOUN)
    Sentence: (0,12)
    Entities:
      "saw" kind="VERB"
      "dog" kind="ANIMAL"
    Dependencies:
      saw -> I (nsubj)
      saw -> dog (obj)
      dog -> a (det)
    """
    cas = Cas(typesystem=typesystem)
    cas.sofa_string = "I saw a dog."

    POS_T = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS")
    SENT_T = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
    ENT_T = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity")
    DEP_T = typesystem.get_type("org.dakoda.syntax.UDependency")

    # POS
    p_I = POS_T(begin=0, end=1, PosValue="PRON")
    p_saw = POS_T(begin=2, end=5, PosValue="VERB")
    p_a = POS_T(begin=6, end=7, PosValue="DET")
    p_dog = POS_T(begin=8, end=11, PosValue="NOUN")
    for p in (p_I, p_saw, p_a, p_dog):
        cas.add(p)

    # Sentence
    sent = SENT_T(begin=0, end=12)
    cas.add(sent)

    # Entities
    e_saw = ENT_T(begin=2, end=5, value="VERB")
    e_dog = ENT_T(begin=8, end=11, value="ANIMAL")
    cas.add(e_saw)
    cas.add(e_dog)

    # Dependencies
    d1 = DEP_T(Governor=p_saw, Dependent=p_I, DependencyType="nsubj")
    d2 = DEP_T(Governor=p_saw, Dependent=p_dog, DependencyType="obj")
    d3 = DEP_T(Governor=p_dog, Dependent=p_a, DependencyType="det")
    cas.add(d1)
    cas.add(d2)
    cas.add(d3)

    return cas


@pytest.fixture()
def cas_two_sentences(typesystem):
    """
    Text: "I saw a dog. It barked."
    S1 as in cas_single_sentence
    S2:
      It: (13,15,PRON)
      barked: (16,22,VERB)
      Sentence: (13,23)
      Dep: barked -> It (nsubj)
    """
    cas = Cas(typesystem=typesystem)
    cas.sofa_string = "I saw a dog. It barked."

    POS_T = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS")
    SENT_T = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
    ENT_T = typesystem.get_type("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity")
    DEP_T = typesystem.get_type("org.dakoda.syntax.UDependency")

    # Sentence 1 and POS
    p_I = POS_T(begin=0, end=1, PosValue="PRON")
    p_saw = POS_T(begin=2, end=5, PosValue="VERB")
    p_a = POS_T(begin=6, end=7, PosValue="DET")
    p_dog = POS_T(begin=8, end=11, PosValue="NOUN")
    cas.add(p_I); cas.add(p_saw); cas.add(p_a); cas.add(p_dog)
    sent1 = SENT_T(begin=0, end=12)
    cas.add(sent1)
    # Entities S1
    cas.add(ENT_T(begin=2, end=5, value="VERB"))
    cas.add(ENT_T(begin=8, end=11, value="ANIMAL"))
    # Dependencies S1
    cas.add(DEP_T(Governor=p_saw, Dependent=p_I, DependencyType="nsubj"))
    cas.add(DEP_T(Governor=p_saw, Dependent=p_dog, DependencyType="obj"))
    cas.add(DEP_T(Governor=p_dog, Dependent=p_a, DependencyType="det"))

    # Sentence 2 and POS
    p_It = POS_T(begin=13, end=15, PosValue="PRON")
    p_barked = POS_T(begin=16, end=22, PosValue="VERB")
    cas.add(p_It); cas.add(p_barked)
    sent2 = SENT_T(begin=13, end=23)
    cas.add(sent2)
    # Entity S2
    cas.add(ENT_T(begin=16, end=22, value="EVENT"))
    # Dependencies S2
    cas.add(DEP_T(Governor=p_barked, Dependent=p_It, DependencyType="nsubj"))

    return cas

