import streamlit as st
import spacy as sp
import util as util
import displacy

cas_file = 'data/hagen.txt.xmi'
ts_file = 'data/TypeSystem.xml'
ts = util.load_typesystem(ts_file)
cas = util.cas_from_string(cas_file, ts)

sofaString="Die Fernuniversität in Hagen (Eigenschreibweise: FernUniversität) ist die erste und einzige staatliche Fernuniversität in Deutschland. Ihr Sitz befindet sich in Hagen in Nordrhein-Westfalen. Nach Angaben des Statistischen Bundesamtes war sie, ohne Berücksichtigung von Akademie- und Weiterbildungsstudierenden, mit über 76.000 Studierenden im Wintersemester 2016/2017[3] die größte deutsche Universität.[4]&#10;&#10;Die Abschlüsse der Fernuniversität sind reguläre Universitätsabschlüsse. Mit dem Sommersemester 2020/2021 wird in keinem Studiengang mehr das Diplom verliehen. Absolventen erhalten somit einen Bachelor- oder Masterabschluss.[5] Alle Fakultäten besitzen das Promotions- und Habilitationsrecht. Außerdem bietet die Fernuniversität in Hagen studienvorbereitende Kurse, Akademiestudium und Teilstudien für die berufliche oder persönliche Weiterbildung. Die Fernuniversität ist Mitglied der European University Association (EUA) und die Studiengänge sind von den beiden Akkreditierungsagenturen FIBAA und AQAS akkreditiert."

#print(sofaString)
#print(cas.sofa_string)

# Spacy
nlp = sp.load('de_core_news_sm')

##### Einkommentiern ===> Fehlerbehebung
#cas.sofa_string = cas.sofa_string.replace('\n\n', '\n')
#####

doc = nlp(cas.sofa_string)
#doc = nlp(sofaString) kein Problem, da &#10;&#10 nicht in \n\n umgewandelt wurde
span_list = []
for sent in doc.sents:
    if sent.start_char >= 500:
        span_list.append(sp.tokens.Span(doc, sent.start, sent.end, "ORG"))
doc.spans["sc"] = span_list


# Streamlit
html = displacy.render(doc, style="span")
print(html)
st.title('Displacy')
st.write(html, unsafe_allow_html=True)
st.title('HTML')
st.write(html, unsafe_allow_html=False)
with open('cas_visualizer/min_working_example.html', 'w') as file:
    file.write(html)
