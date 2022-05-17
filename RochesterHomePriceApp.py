# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:19:05 2022

@author: T430s
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# loading the saved model for Home Price(random forest model was used)
with open('model_RochesterHome', 'rb') as f1: # rb = read the binary file
    model_price = pickle.load(f1)

# loading the saved model for Tax(random forest model was used)  
with open('model_RochesterHomeTax', 'rb') as f2: 
    model_tax = pickle.load(f2)
   
with open('df_RochesterHomePrice', 'rb') as f3: 
    df = pickle.load(f3)


st.write("""
## Rochester, NY House Price/Tax Prediction 
This app predicts the **Rochester,NY Region House Price by --A. Maharjan**
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

#def user_input_features():
District = st.sidebar.selectbox('Select the District for Price/Tax Prediction', ('gates','greece','chili','penfield','pittsford','webster','brighton','henrietta','East Rochester','Rochester City'))
Bedroom = st.sidebar.slider('Bedroom', 1,10,3)
Bathroom = st.sidebar.slider('Bathroom', 1,8)
Area = st.sidebar.slider('Area', 100,8000,1850)
Age = st.sidebar.slider('How Old is the Home?', 0,200,55)
    
data = {'District': District,
            'Bedroom': Bedroom,
            'Bathroom': Bathroom,
            'Area': Area,
            'Age': Age,
            }
features_price = pd.DataFrame(data, index=[0])


# Main Panel
# Print specified input parameters
st.header('Specified Input parameters')
st.write(features_price)
st.write('---')

# Apply Model to Make Prediction
prediction_price = model_price.predict(features_price)    
features_tax = features_price
features_tax['Price'] = prediction_price 
prediction_tax = model_tax.predict(features_price)

if st.button('Predict HomePrice'):
    
    st.write('Extimated Home Price is $','%.2f' % prediction_price)
    st.write('Estimated Tax on this home is $','%.2f' % prediction_tax)


#df_input = user_input_features()

# for ANALYTICS
st.sidebar.header('Analytics: Specify District')
district = st.sidebar.selectbox('Select District for Analytics', ('gates','greece','chili','penfield','pittsford','webster','brighton','henrietta','East Rochester','Rochester City'))
if st.sidebar.button('Histogram/BoxPlot of Price'):
    df_district = df[df.District == district]
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(df_district,x='Price',bins = 10)
    plt.xlabel('Price')
    plt.title(district)
    st.pyplot()
    plt.boxplot(df_district['Price'])
    plt.xlabel('Price')
    plt.title(district)
    plt.show()
    st.pyplot()
    st.write(df_district.describe())
    
    


if st.sidebar.button('Histogram/BoxPlot of Tax'):
    df_district = df[df.District == district]
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(df_district,x='Tax',bins=10)
    plt.xlabel('Tax')
    plt.title(district)
    st.pyplot()
    plt.boxplot(df_district['Tax'])
    plt.xlabel('Tax')
    plt.title(district)
    plt.show()
    st.pyplot()
    st.write(df_district.describe())
    
    
    
# summary sns box plot for all district
st.sidebar.header('Overall Price Distribution')
if st.sidebar.button('Overall Price Distribution by District'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    ax = sns.boxplot(x='District',y='Price',data=df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    st.pyplot()
if st.sidebar.button('Overall Price Distribution(Histogram)'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(x='Price',data=df)
    st.pyplot()
    st.write(df.Price.describe())
    
    
    
    



st.sidebar.header('Overall Tax Distribution')
if st.sidebar.button('Overall Tax Distribution by District'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    ax = sns.boxplot(x='District',y='Tax',data=df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    st.pyplot()
    
if st.sidebar.button('Overall Tax Distribution(Histogram)'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    sns.histplot(x='Tax',data=df)
    st.pyplot()
    st.write(df.Tax.describe())
    
    





# pair plot between the price vs other variables
st.sidebar.header('Correlation')
if st.sidebar.button('Pair/Correlation Plot'):
    
    st.set_option('deprecation.showPyplotGlobalUse', False) # not to print the error message
    #cars_num=cars.select_dtypes(include=['float64','int64','int32'])
    sns.pairplot(df,x_vars = ['Bedroom','Bathroom','Area','Age','Tax'],
             y_vars=['Price'], kind='reg', plot_kws={'line_kws':{'color':'red'}})
    
    #fig = plt.figure(figsize=(12,8)) 
    st.pyplot()













