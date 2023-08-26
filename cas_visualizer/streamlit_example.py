import numpy as np
import pandas as pd
import streamlit as st
from cassis import *
from operator import itemgetter

# st.set_page_config(layout="wide")
st.title("CAS Visualizer")

# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
# load and read the cas + typesystem, create a list representation for further processing
def cas_read_preprocessing():

    # --------------------------------------------------------------
    # static stuff, not needed for every cas but for the example cas,
    # was formerly a parameter, but for this showcase its hardwired
    pathSen = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    pathTok = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
    pathPos = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
    pathFreq = "org.lift.type.Frequency"

    with open('data/TypeSystem.xml', 'rb') as f:
        typesys = load_typesystem(f)

    with open('data/hagen.txt.xmi', 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesys)

    # --------------------------------------------------------------



    possibleTypes = ["noType"]
    sofaString = ""
    tokenText = []
    tokenType = []
    tokenBegin = []
    toBeSorted = []
    allToken = []


    for sentence in cas.select(pathSen):  # sentence

        for t in cas.select_covered(pathTok, sentence):
            allToken.append([t.begin, t.get_covered_text(), "noType"])

        for token in cas.select_covered(pathPos, sentence):
            if token.coarseValue not in possibleTypes:
                possibleTypes.append(token.coarseValue)
            toBeSorted.append([token.begin, token.get_covered_text(), token.coarseValue])
            sortedArray = sorted(toBeSorted, key=itemgetter(0))

        for tokA in allToken:
            for tokB in sortedArray:
                if tokA[0] == tokB[0]:
                    tokA[2] = tokB[2]

    for sortedItem in allToken:
        tokenBegin.append(sortedItem[0])
        tokenText.append(sortedItem[1])
        tokenType.append(sortedItem[2])

    for word in tokenText:
        sofaString = sofaString + word + " "
    return possibleTypes, allToken, sofaString


# -----------------------------------------------------------------------------------------
# table
def visualize_cas_table():
    a = cas_read_preprocessing()[1]
    df = pd.DataFrame(np.asarray(a), columns=['start', 'token', 'type'])
    st.table(df.drop(['start'], axis=1))


# -----------------------------------------------------------------------------------------
# span
def visualize_cas_span():
    typeArray, sortedArray, sofaString = cas_read_preprocessing()
    assignColors_and_multiselect(typeArray, sortedArray, sofaString)


# quick method to wrap the html part around a token (assign background color)
# can be modified like normal HTML
def spanWrapper(color, token, underline=None, subscript=None, tooltip=None):
    #subscript = "dummy"
    #underline = "dummy"
    #tooltip = "dummy"
    subscript_msg = ""
    if subscript is not None:
        subscript_msg = "<sub style=\"color:grey\">"+subscript+"</sub>"
    underline_msg = ""
    if underline is not None:
        underline_msg = "; text-decoration: underline;text-decoration-thickness: 2px; text-underline-offset:0.1cm; text-decoration-color:" + str(color)
    tooltip_msg = ""
    if tooltip is not None:
        tooltip_msg = " title=\""+tooltip+"\" "
    return"<span " +tooltip_msg+ "style=\"border-radius: 25px;padding-left:5px; padding-right:5px; background-color: " +str(color)+underline_msg+"\"/>"+str(token)+ subscript_msg +"</span> "


# color matching and multiselect
def assignColors_and_multiselect(typeArray, sortedArray, sofaString):
    # get all types
    alreadySeen = []
    for t in typeArray:
        if t not in alreadySeen:
            alreadySeen.append(t)

    alreadySeenWithoutNoType = alreadySeen.copy()
    alreadySeenWithoutNoType.remove('noType')
    currentType = st.multiselect("Select Type: ", alreadySeenWithoutNoType, alreadySeenWithoutNoType)

    # different color schemes, feel free to create your own, keep possible dark mode users in mind
    availableColors3 = ["coral", "chartreuse", "orchid", "gold", "cornflowerblue", "lightseagreen",
                        "mediumpurple", "springgreen", "indianred", "hotpink", "darkorange", "palevioletred",
                        "darkkhaki", "greenyellow", "palegreen"]
    availableColors = ["orangered", "orange", "plum", "palegreen", "mediumseagreen", "lightseagreen",
                       "steelblue", "skyblue", "navajowhite", "mediumpurple", "rosybrown", "silver", "gray",
                       "paleturquoise"]
    availableColors2 = ["lightcoral", "pink", "palevioletred", "lightsalmon", "gold", "lightskyblue", "lavender",
                        "plum",
                        "palegreen", "mediumaquamarine", "darkseagreen", "paleturquoise", "lightsteelblue", "rosybrown",
                        "gainsboro"]

    typeWithColor = []  # for the type-color display above the text
    textWithColor = []  # for the actual tokens that get colored

    # upper limit not really needed for this example as cas is hardwired into the code
    if len(currentType) > 13 or len(currentType) == 0:
        st.write("Nothing selected! Or too much (max 13).")
        st.write(sofaString)
    else:
        # assign each type a unique color
        for types in alreadySeen:
            typeWithColor.append([types, availableColors[alreadySeen.index(types)]])

        for elements in sortedArray:

            # match each token to its respective type and assign corresponding color
            for colorPair in typeWithColor:
                if colorPair[0] == elements[2]:
                    textWithColor.append([elements[1], colorPair[1], elements[2], elements[0]])

        # these will be the strings that get displayed in the app
        finalString = ""
        finalTypeString = ""

        # finalize the types string (each used type with its color)
        for types in typeWithColor:
            if types[0] in currentType and types[0] != "noType":
                finalTypeString = finalTypeString + spanWrapper(types[1], types[0])

        # finalize text (each token with a type gets a color)
        # the if-else labyrinth is for the different places a token can be in a text (punctucation, whitespaces, etc.)
        for triple in textWithColor:
            # filter only selected types and exclude noType
            if triple[2] != "noType" and triple[2] in currentType:
                # if token is punctuation, no whitespaces needed
                if triple[2] == "PUNCT":
                    finalString = finalString + spanWrapper(triple[1], triple[0])
                else:
                    # beginning of text
                    if finalString == "":
                        finalString = finalString + spanWrapper(triple[1], triple[0])
                    else:
                        # token after another token
                        finalString = finalString + " " + spanWrapper(triple[1], triple[0])
            else:
                # for noType and not selected types
                if finalString == "":
                    finalString = triple[0]
                else:
                    finalString = finalString + " " + triple[0]

        # sounds dangerous, but it works ;)
        # and is commonly used in other streamlit components to mess around with texts
        st.write(finalTypeString, unsafe_allow_html=True)
        st.write("---------------------")
        st.write(finalString, unsafe_allow_html=True)


# -----------------------------------------------------------------------------------------
# choose your visualization style
visualize_cas_span()
visualize_cas_table()
