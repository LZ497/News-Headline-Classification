#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 06:06:15 2021

@author: galeliu
"""

import streamlit as st
from streamlit import bootstrap
import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


pred_df = pd.read_csv("/Users/galeliu/Downloads/pred.csv")
pred_rf_proba = pd.read_csv("/Users/galeliu/Downloads/pred_rf_proba.csv")
pred_nb_proba = pd.read_csv("/Users/galeliu/Downloads/pred_nb_proba.csv")
#pred_df = pred_df.groupby('Category').apply(lambda x: x.sample(frac=0.01))


st.write("""
# News Headline Classification App
This app predicts the **News Headline** category!
""")

def news_headline_changed():
    new_news = st.info(f"news_headline_changed: {st.session_state.selectbox}")
    return new_news

st.subheader('News Headline')
selectbox = st.selectbox("Select a news headline", pred_df['News Headline'], key="selectbox", on_change=news_headline_changed)
st.write('The News Headline selected is: ', selectbox)

st.markdown('##')

final_pred_rf=pred_df['Category_rf'][pred_df.index[pred_df['News Headline'] == selectbox]]
st.subheader('Random Forest Prediction')
st.write(str(final_pred_rf.values))

st.subheader('Random Forest Prediction Probability')
st.write(pred_rf_proba.iloc[pred_df.index[pred_df['News Headline'] == selectbox]])

st.markdown('##')

final_pred_nb=pred_df['Category_nb'][pred_df.index[pred_df['News Headline'] == selectbox]]
st.subheader('Multinomial Naive Bayes Prediction')
st.write(str(final_pred_nb.values))

st.subheader('Multinomial Naive Bayes Prediction Probability')
st.write(pred_nb_proba.iloc[pred_df.index[pred_df['News Headline'] == selectbox]])

st.markdown('##')

final_pred_svm=pred_df['Category_svm'][pred_df.index[pred_df['News Headline'] == selectbox]]
st.subheader('SVM Prediction')
st.write(str(final_pred_svm.values))

################# RUN THE NEXT LINE IN TERMINAL #################
################# streamlit run final_web.py ####################
#################################################################

#args=[]
#st.bootstrap.run("/Users/galeliu/Downloads/streamlit_webpage.py", '', args, flag_options={})
#st.stop()

