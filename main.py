import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import xgboost as xgb
import numpy as np

st.write("""
# StoCast
## Tesla ($TSLA) Stock Movement Prediction App
### CSE420 Honours Project - OCdt Joon Choi, 29793
         
This app predicts how $TSLA's stock price will have moved 5 days from now.
The data from this app shall only be used as reference ONLY. The creator is not liable for any loss of any kind.
""")
st.write('---')
