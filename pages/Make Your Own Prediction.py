import streamlit as st
from streamlit_shap import st_shap
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import xgboost as xgb
import numpy as np

# Todo: Add reddit sentences

st.set_page_config(page_title="Custom Check", page_icon="📈")

grad = pd.read_csv(r'C:/Users/jchoi/Desktop/School/Year 4 Fall/CSE420/App/v6_merge.csv')

columns_to_convert = ['Close', 'Next_Close','Long_Change',
       'Volume', 'Open', 'Gap', 'High', 'Low', 'EMA_26', 'EMA_12', 'MACD',
       'MACD_9', 'RSI','count_wsb', 'upvotes_wsb', 'bullish_wsb',
       'neutral_wsb', 'bearish_wsb','count_stocks', 'upvotes_stocks', 'bullish_stocks',
       'neutral_stocks', 'bearish_stocks']
grad[columns_to_convert] = grad[columns_to_convert].apply(pd.to_numeric, errors='coerce')

X_non = grad.drop(columns=['Next_Close','Positive_Spike', 'Long_Change','Gap']) 
X = X_non.select_dtypes(include='number')
print("This is X")
print(X)
y = grad.loc[:, 'Long_Change']  # target

# Sidebar
# Header of Specify Input Parameters
st.header("How will $TSLA's price change in 5 days?")
st.write("Adjust the sliders on the left to input the data.")
st.write('---')

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Close = st.sidebar.slider('Closing Price Change from Yesterday', X.Close.min(), X.Close.max(), X.Close.mean())
    Volume = st.sidebar.slider('Volume Change from Yesterday', X.Volume.min(), X.Volume.max(), X.Volume.mean())
    Open = st.sidebar.slider('Opening Price Change from Yesterday', X.Open.min(), X.Open.max(), X.Open.mean())
    High = st.sidebar.slider('High Price Change from Yesterday', X.High.min(), X.High.max(), X.High.mean())
    Low = st.sidebar.slider('Low Price Change from Yesterday', X.Low.min(), X.Low.max(), X.Low.mean())
    EMA_26 = st.sidebar.slider('26-Day Exponential Moving Average Change from Yesterday', X.EMA_26.min(), X.EMA_26.max(), X.EMA_26.mean())
    EMA_12 = st.sidebar.slider('12-Day Exponential Moving Average Change from Yesterday', X.EMA_12.min(), X.EMA_12.max(), X.EMA_12.mean())
    MACD = st.sidebar.slider('MACD Change from Yesterday', X.MACD.min(), X.MACD.max(), X.MACD.mean())
    MACD_9 = st.sidebar.slider('9-day MACD Change from Yesterday', X.MACD_9.min(), X.MACD_9.max(), X.MACD_9.mean())
    # MACD_momentum = st.sidebar.slider('MACD Momentum', X.MACD_momentum.min(), X.MACD_momentum.max())
    MACD_signal = st.sidebar.slider('MACD Signal', X.MACD_signal.min(), X.MACD_signal.max())
    RSI = st.sidebar.slider('Relative Strength Indicator Value', X.RSI.min(), X.RSI.max(), X.RSI.mean())
    # RSI_overbought = st.sidebar.slider('RSI Over 65', X.RSI_overbought.min(), X.RSI_overbought.max())
    # RSI_oversold = st.sidebar.slider('RSI Under 35', X.RSI_oversold.min(), X.RSI_oversold.max())
    RSI_signal = st.sidebar.slider('RSI Signal', X.RSI_overbought.min(), X.RSI_overbought.max())
    count_wsb = st.sidebar.slider('Number of Submissions & Comments on r/WSB', X.count_wsb.min(), X.count_wsb.max(), X.count_wsb.mean())
    upvotes_wsb = st.sidebar.slider('Average Number of Upvotes on r/WSB', X.upvotes_wsb.min(), X.upvotes_wsb.max(), X.upvotes_wsb.mean())
    # bullish_wsb = st.sidebar.slider('Bullish Sentiment in r/WSB', X.bullish_wsb.min(), X.bullish_wsb.max(), X.bullish_wsb.mean())
    # neutral_wsb = st.sidebar.slider('Neutral Sentiment in r/WSB', X.neutral_wsb.min(), X.neutral_wsb.max(), X.neutral_wsb.mean())
    # bearish_wsb = st.sidebar.slider('Bearish Sentiment in r/WSB', X.bearish_wsb.min(), X.bearish_wsb.max(), X.bearish_wsb.mean())
    wsb_sentiment = st.sidebar.slider('Sentiment on r/WSB', -1, 1, 0)
    count_stocks = st.sidebar.slider('Number of Submissions & Comments on r/Stocks', X.count_stocks.min(), X.count_stocks.max(), X.count_stocks.mean())
    upvotes_stocks = st.sidebar.slider('Average Number of Upvotes on r/Stocks', X.upvotes_stocks.min(), X.upvotes_stocks.max(), X.upvotes_stocks.mean())
    stocks_sentiment = st.sidebar.slider('Sentiment on r/Stocks', -1, 1, 0)
    # bullish_stocks = st.sidebar.slider('Bullish Sentiment in r/Stocks', X.bullish_stocks.min(), X.bullish_stocks.max(), X.bullish_stocks.mean())
    # neutral_stocks = st.sidebar.slider('Neutral Sentiment in r/Stocks', X.neutral_stocks.min(), X.neutral_stocks.max(), X.neutral_stocks.mean())
    # bearish_stocks = st.sidebar.slider('Bearish Sentiment in r/Stocks', X.bearish_stocks.min(), X.bearish_stocks.max(), X.bearish_stocks.mean())
    # sliders don't account for the sentiment scores properly so this is an estimate
    if MACD > MACD_9:
        MACD_momentum = 1
    else:
        MACD_momentum = 0

    RSI_overbought = 0
    RSI_oversold = 0
    if RSI > 65:
        RSI_overbought = 1
    elif RSI < 35:
        RSI_oversold = 1

    if wsb_sentiment == 1:
        bullish_wsb = 0.6
        neutral_wsb = 0.2
        bearish_wsb = 0.2
    elif wsb_sentiment == 0:
        bullish_wsb = 0.2
        neutral_wsb = 0.6
        bearish_wsb = 0.2
    else:
        bullish_wsb = 0.2
        neutral_wsb = 0.2
        bearish_wsb = 0.6

    if stocks_sentiment == 1:
        bullish_stocks = 0.6
        neutral_stocks = 0.2
        bearish_stocks = 0.2
    elif stocks_sentiment == 0:
        bullish_stocks = 0.2
        neutral_stocks = 0.6
        bearish_stocks = 0.2
    else:
        bullish_stocks = 0.2
        neutral_stocks = 0.2
        bearish_stocks = 0.6

    data = {'Close': Close,
            'Volume': Volume,
            'Open': Open,
            'High': High,
            'Low': Low,
            'EMA_26': EMA_26,
            'EMA_12': EMA_12,
            'MACD': MACD,
            'MACD_9': MACD_9,
            'MACD_momentum': MACD_momentum,
            'MACD_signal': MACD_signal,
            'RSI': RSI,
            'RSI_overbought': RSI_overbought,
            'RSI_oversold': RSI_oversold,
            'RSI_signal': RSI_signal,
            'count_wsb': count_wsb,
            'upvotes_wsb': upvotes_wsb,
            'bullish_wsb': bullish_wsb,
            'neutral_wsb': neutral_wsb,
            'bearish_wsb': bearish_wsb,
            'count_stocks': count_stocks,
            'upvotes_stocks': upvotes_stocks,
            'bullish_stocks': bullish_stocks,
            'neutral_stocks': neutral_stocks,
            'bearish_stocks': bearish_stocks,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.dataframe(df, hide_index=True)
st.write('---')

# Build Regression Model
model = xgb.Booster()
model.load_model('xgboost_model3.model')

gpu_id = 0  # Specify the GPU ID
if hasattr(model, 'gpu_id'):
    model.set_param({"gpu_id": gpu_id})

dmatrix = xgb.DMatrix(df)
# Apply Model to Make Prediction
prediction = model.predict(dmatrix)

st.header('StoCast Says:')

if prediction[0] > 5:
    st.markdown(f'<h1 style="color:darkgreen;">+{prediction[0]:.3}%</h1>', unsafe_allow_html=True)
    st.write(f'Get on the wave! StoCast thinks there is a significant price jump coming in 5 days.')
elif prediction[0] < 5 and prediction[0] >= 0:
    st.markdown(f'<h1 style="color:lightgreen;">+{prediction[0]:.3}%</h1>', unsafe_allow_html=True)
    st.write(f'Good stuff! StoCast thinks the price is going UP in 5 days!')
elif prediction[0] < 0:
    st.markdown(f'<h1 style="color:red;">{prediction[0]:.3}%</h1>', unsafe_allow_html=True)
    st.write(f'Brace yourself! StoCast thinks the price is going DOWN in 5 days!')

st.write('---')

st.set_option('deprecation.showPyplotGlobalUse', False)

button_clicked = st.button("Explain Prediction")

# Check if the button is clicked
if button_clicked:
    st.header('Why did StoCast make this prediction?')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    shap_values_big = explainer.shap_values(X)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], df.iloc[0,:]), height=200, width=800)
    st_shap(shap.decision_plot(explainer.expected_value, shap_values[0,:], df.iloc[0,:]), height=500, width=800)

    st.header("Feature Importance in StoCast:")
    st_shap(shap.summary_plot(shap_values_big, X))

    explainer1 = shap.Explainer(model, X)
    shap_values1 = explainer1(X)

    st.write('---')