import pandas as pd
import numpy as np
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier
classifier = TextClassifier.load('en-sentiment')



st.image("coeai.png", width=120)
st.title("Profanity filter")
x = st.text_input("Text")
if st.button("Check"):
    sentence = Sentence(x)
    classifier.predict(sentence)
    result = sentence.labels
    st.success("The oput is: {}".format(result))
