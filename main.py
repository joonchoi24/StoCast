import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import xgboost as xgb
import numpy as np

st.title("Welcome to...")
st.image('gaga.png')
st.write("""
## A Transparent and Robust Stock Movement Forecasting Platform with Explainable Ensemble Learning.
### CSE420 Honours Project - OCdt Joon Choi, 29793
         
This app predicts how $TSLA's stock price will have moved 5 days from now, using financial indicators and social media trends.
         Click on any of the tabs on the left to get started!

         
*The data from this app shall only be used as reference ONLY. The creator is not liable for any loss of any kind.*
""")
st.write('---')
