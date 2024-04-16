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
st.image('gaga3_real.png')
st.write("""
## A Transparent and Robust Stock Movement Forecasting Platform with Explainable Ensemble Learning.
### CSE420 Honours Project - OCdt Joon Choi, 29793
         
This app predicts how $TSLA's stock price will have moved 5 days from now, using financial indicators and social media trends.

Click on any of the tabs on the left to get started!
     

""")
st.write('---')
st.write("""Features:
* Predict **$TSLA's** stock price 5 days from any given day.
* Use the default parameters, or enter your own - you can play around with over **18 different parameters**!
* Understand the predictions fully and clearly with **Explainable AI**: every prediction is justified and communicated clearly to you with SHAP (SHapley Additive exPlanations) plots. This helps you understand WHY StoCast made such a prediction.
* View StoCast's performance on a custom range of historical data, and how much you could have earned by following StoCast's advice.
* Compare how various strategies would have played out, depending on your activity and risk tolerance.
* Interact with the clean and intuitive UI.
         
***The data from this app shall only be used as reference ONLY. The creator is not liable for any loss of any kind.***
""")