import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import xgboost as xgb
import numpy as np

st.set_page_config(page_title="Historical Check", page_icon="📈")

grad = pd.read_csv(r'C:/Users/jchoi/Desktop/School/Year 4 Fall/CSE420/App/v6_merge.csv')
# Build Regression Model
model = xgb.Booster()
model.load_model('xgboost_model2.model')

gpu_id = 0  # Specify the GPU ID
if hasattr(model, 'gpu_id'):
    model.set_param({"gpu_id": gpu_id})

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

st.write('---')
st.header('Historical Test')
st.write('Check for yourself how StoCast does on historical data.')
# def load_data():
#     # Load your DataFrame here
#     return pd.read_csv('your_data.csv')  # Replace with your actual DataFrame loading code

# grad_test = load_data()

grad.set_index('Date', inplace=True)
# Display the DataFrame for selection
selected_index = st.selectbox('Select a row from the DataFrame:', grad.index)

selected_row = grad.loc[[selected_index]]
unwanted_columns = ['Next_Close', 'Positive_Spike', 'Long_Change','Gap']  # Replace with the names of the columns you want to drop
selected_row_filtered = selected_row.drop(columns=unwanted_columns)
selected_row_dropped = selected_row_filtered.select_dtypes(include='number')

dmatrix_sample = xgb.DMatrix(selected_row_dropped)
sample_prediction = model.predict(dmatrix_sample)

# Show the selected row
st.write('Selected Row:')
st.write(selected_row_filtered)

table_data = {
    'StoCast Prediction': [sample_prediction[0]],
    'Actual Movement': [selected_row['Long_Change'][0]]
}
table_df = pd.DataFrame(table_data)

st.header('StoCast Prediction vs. Actual Movement')
st.dataframe(table_df, hide_index=True)
#st.write(table_df, hide_index=True)
#st.table(table_df, hide_index=True)

explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)



shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[3], X.iloc[3])

shap.dependence_plot(0, shap_values, X)
# Provide a random example for testing
# random_index = np.random.randint(len(grad_copy))
# random_example = grad_copy.loc[random_index]

# # Show the random example
# st.write('Random Example for Testing:')
# st.write(random_example)