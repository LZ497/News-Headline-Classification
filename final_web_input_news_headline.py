#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 22:56:33 2021

@author: galeliu
"""

import streamlit as st
from streamlit import bootstrap
import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB

import pickle

# load the model
text_clf_nb = pickle.load(open("/Users/galeliu/Downloads/text_clf_nb.pkl", "rb"))

st.write("""
# News Headline Classification App
This app predicts the **News Headline** category!
""")



st.subheader('News Headline')
string1 = st.text_input('Enter News Headline:', 'Omicron Is Here. Should You Cancel Your Trip?')
st.write('The current News Headline is: ', string1)

new_text_df = pd.DataFrame([string1], columns=["text"])
new_predicted = text_clf_nb.predict(new_text_df["text"])

new_prediction_proba = text_clf_nb.predict_proba(new_text_df["text"])

st.subheader('Prediction')
#st.write(predicted[1])
st.write(str(new_predicted))

pred_all = pd.read_csv("/Users/galeliu/Downloads/pred.csv")
pred_all_colname = sorted(pred_all.Category_nb.unique())

pred_all_proba = pd.DataFrame(new_prediction_proba, columns=pred_all_colname)

st.subheader('Prediction Probability')
st.write(pred_all_proba)
#st.write(new_prediction_proba)
#args=[]
#st.bootstrap.run("/Users/galeliu/Downloads/streamlit_webpage.py", '', args, flag_options={})
#st.stop()

