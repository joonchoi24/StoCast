import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import xgboost as xgb
import numpy as np

# Todo: historical range, pick and choose and display social media data

st.set_page_config(page_title="Historical Check", page_icon="📈")

grad = pd.read_csv(r'C:/Users/jchoi/Desktop/School/Year 4 Fall/CSE420/App/v6_merge_test_v4.csv')
model = xgb.Booster()
model.load_model('xgboost_model4.model')

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

grad.set_index('Date', inplace=True)
grad.index = pd.to_datetime(grad.index)

start_date, end_date = st.date_input('Select date range:', [grad.index.min(), grad.index.max()])
selected_rows_multi = grad.loc[start_date:end_date]
unwanted_columns = ['Next_Close', 'Positive_Spike', 'Long_Change','Gap']
selected_row_filtered_multi = selected_rows_multi.drop(columns=unwanted_columns)
selected_row_dropped_multi = selected_row_filtered_multi.select_dtypes(include='number')
dmatrix_sample_multi = xgb.DMatrix(selected_row_dropped_multi)
sample_prediction_multi = model.predict(dmatrix_sample_multi)
table_data_multi = []

# Iterate over the indices
for i in range(len(sample_prediction_multi)):
    row_data = {
        'StoCast Prediction': sample_prediction_multi[i],
        'Actual Movement': selected_rows_multi['Long_Change'][i]
    }
    table_data_multi.append(row_data)

# Create a DataFrame from the list of dictionaries
table_df_multi = pd.DataFrame(table_data_multi, index=selected_rows_multi.index.strftime('%Y-%m-%d'))
st.dataframe(table_df_multi)

def mda(df):
    # Merge the Series based on day number

    # Calculate the sign of the change for predicted and actual values
    df['actual_sign'] = df['Actual Movement'].apply(lambda x: 1 if x >= 0 else -1)
    df['pred_sign'] = df['StoCast Prediction'].apply(lambda x: 1 if x >= 0 else -1)

    # Calculate accuracy
    correct_predictions = (df['pred_sign'] == df['actual_sign']).sum()

    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print(f"Accuracy: {accuracy}")
    print(df)
    return accuracy

def profit(df):
    # Calculate the sign of the change for predicted and actual values
    df['actual_sign'] = df['Actual Movement'].apply(lambda x: 1 if x >= 0 else -1)
    df['pred_sign'] = df['StoCast Prediction'].apply(lambda x: 1 if x >= 0 else -1)

    init = 1000
    prod = init  # Value if simply buying at every positive day and selling after 5 days
    prod_prop = init  # Value if proportionally buying based on percentage, taking the model's prediction as its confidence, 25% of money if < 1%, 50% if < 3%, 75% if 5%, 100% if greater
    prod_absprop = init  # Similar but constant dollars, $250 - $500 - $750 - $1000
    prod_oppo = init  # prod_prop but opposite proportions
    values = {'prod': [], 'prod_prop': [], 'prod_absprop': [], 'date': []}
    for index, row in df.iloc[::5].iterrows(): # every 5 days
    #for index, row in df.iterrows():
        if row['pred_sign'] == 1:
            prod = prod * ((row['Actual Movement'] / 100) + 1)  # Multiplies the init remaining by the actual change of price
            if row['StoCast Prediction'] < 1:
                prod_prop = (prod_prop - prod_prop * 0.25) + ((prod_prop * 0.25) * ((row['Actual Movement'] / 100) + 1))
                if prod_absprop >= 250:
                    prod_absprop = (prod_absprop - 250) + (250 * ((row['Actual Movement'] / 100) + 1))
            elif row['StoCast Prediction'] < 3:
                prod_prop = (prod_prop - prod_prop * 0.5) + ((prod_prop * 0.5) * ((row['Actual Movement'] / 100) + 1))
                if prod_absprop >= 500:
                    prod_absprop = (prod_absprop - 500) + (500 * ((row['Actual Movement'] / 100) + 1))
            elif row['StoCast Prediction'] < 5:
                prod_prop = (prod_prop - prod_prop * 0.75) + ((prod_prop * 0.75) * ((row['Actual Movement'] / 100) + 1))
                if prod_absprop >= 750:
                    prod_absprop = (prod_absprop - 750) + (750 * ((row['Actual Movement'] / 100) + 1))
            else:
                prod_prop = prod_prop * ((row['Actual Movement'] / 100) + 1)
                if prod_absprop >= 1000:
                    prod_absprop = (prod_absprop - 1000) + (1000 * ((row['Actual Movement'] / 100) + 1))
        values['prod'].append(prod)
        values['prod_prop'].append(prod_prop)
        values['prod_absprop'].append(prod_absprop)
        values['date'].append(index)

    # Generate graph
    chart_data = pd.DataFrame(values, index=pd.to_datetime(values['date']))

    # Plot the chart
    st.line_chart(chart_data[['prod', 'prod_prop', 'prod_absprop']])
    # Calculate and display results
    print(f"Always reinvesting fully, $1000 became: {prod}")
    print(f"Reinvesting 25~100% proportionally, $1000 became: {prod_prop}")
    print(f"Reinvesting $250~1000 proportionally, $1000 became: {prod_absprop}")
    print(f"Full: {((prod-init)/init)*100:.1f}% change within {len(df)} days")
    print(f"Percentage proportional: {((prod_prop-init)/init)*100:.1f}% change within {len(df)} days")
    print(f"Absolute proportional: {((prod_absprop-init)/init)*100:.1f}% change within {len(df)} days")
    best = max(prod, prod_prop, prod_absprop)
    return ((best-init)/init)*100

best_dollar = profit(table_df_multi)

st.write(f'Directional Accuracy: {(mda(table_df_multi))*100:.1f}%')
st.write(f'Your \$1000 could have become as much as: ')
if best_dollar > 0:
    st.markdown(f'<h1 style="color:green;">${(1000*((best_dollar/100)+1)):.2f} ({best_dollar:.2f}%)</h1>', unsafe_allow_html=True)
else:
    st.markdown(f'<h1 style="color:red;">${(1000*((best_dollar/100)+1)):.2f} ({best_dollar:.2f}%)</h1>', unsafe_allow_html=True)
st.write(f'within **{len(table_df_multi)}** days.')

st.write('Note: ')
st.write('*prod* assumes full reinvestment of any amount remaining after gaining or losing money from the last trade.')
st.write('*prod_prop* takes the model''s output as its confidence level, investing 25\% of funds for growth 1\% or less, 50\% for 3\% or less, 75\% for 5\% or less, and 100\% for anything greater.')
st.write('*prod_absprop* is similar, except absolute amounts invested instead of percentages, reducing the risk. In the same brackets, \$250, \$500, \$750, or \$1000 is invested.')
# Multiple dates
#selected_indices = st.multiselect('Select rows from the DataFrame:', grad.index)

# if single_button:
#     # Single date
#     # Display the DataFrame for selection
#     selected_index = st.selectbox('Select a row from the DataFrame:', grad.index)

#     selected_row = grad.loc[[selected_index]]
#     selected_row_filtered = selected_row.drop(columns=unwanted_columns)
#     selected_row_dropped = selected_row_filtered.select_dtypes(include='number')

#     dmatrix_sample = xgb.DMatrix(selected_row_dropped)
#     sample_prediction = model.predict(dmatrix_sample)

#     # Show the selected row
#     st.write('Selected Row:')
#     st.write(selected_row_filtered)

#     table_data = {
#         'StoCast Prediction': [sample_prediction[0]],
#         'Actual Movement': [selected_row['Long_Change'][0]]
#     }
#     table_df = pd.DataFrame(table_data)

#     st.header('StoCast Prediction vs. Actual Movement')
#     st.dataframe(table_df, hide_index=True)
#     #st.write(table_df, hide_index=True)
#     #st.table(table_df, hide_index=True)

#     explainer = shap.Explainer(model)
#     shap_values = explainer.shap_values(X)



#     shap.initjs()
#     shap.force_plot(explainer.expected_value, shap_values[3], X.iloc[3])

#     shap.dependence_plot(0, shap_values, X)
#     # Provide a random example for testing
#     # random_index = np.random.randint(len(grad_copy))
#     # random_example = grad_copy.loc[random_index]

#     # # Show the random example
#     # st.write('Random Example for Testing:')
#     # st.write(random_example)
