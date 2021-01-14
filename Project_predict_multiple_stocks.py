#Goal: to give recommendation for stock day trading.
#predicts tomorrow's price for all stocks and shows stocks
#with most profit and most loss.


import pandas_datareader as web
from glob import glob
from datetime import datetime
import os
import pandas_datareader as web
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import tensorflow as tf
from datetime import date, timedelta
import time

#Puts stock tickers in a list
def get_name(file):
    base = os.path.basename(file)
    os.path.splitext(base)
    stock_name = os.path.splitext(base)[0].replace('.us','')
    return stock_name
    
data_files = glob('C:/Users/tahaa/Desktop/python/AI_Project/Stocks/*.txt')
stock_names = [get_name(data_file) for data_file in data_files]

#creates the dataframe
def create_df():
    
    li = []
    last_60_days = list(range(0, 60))
    today = str(date.today())
    
    for i in range(0,len(stock_names)):
        
        stock_name = stock_names[i]
        try:
            df = web.DataReader(stock_name, data_source='yahoo', end=today)
            new_df = df.filter(['Close'])[-60:]
            new_df.columns = [stock_name]
            new_df.index = last_60_days
            new_df = new_df.T
            print(i, end=' ')
            li.append(new_df)
        except:
            continue 
            
    frame = pd.concat(li)
    frame.to_csv('C:/Users/tahaa/Desktop/example.csv', index=True)

create_df()

dff = pd.read_csv('C:/Users/tahaa/Desktop/example.csv')

stock_names_updated = dff['Name'].tolist()


numb = []
for i in range (60):
    numb.append(str(i))
    
#Predict for each row of dataframe(for each stock) 
def predict(row):
    
    stock_name = row['Name']

    last_60_days = row[numb].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(last_60_days)
     
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    #model = models[stock_names_updated.index(stock_name)]
    model = keras.models.load_model('C:/Users/tahaa/Desktop/python/AI_Project/Models/' + stock_name + '.us_model')
    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    if pred[0][0] <= 0 :
        return
    return(pred[0][0])

#adds 'pred' and 'profit' columns to the dataframe
dff['pred'] = dff.apply(lambda row: predict(row), axis=1)
dff['profit%'] = ((dff['pred'] - dff['59'])/dff['59'])*100


#Returns top profit stocks 
def most_profit(frame, N):
    return frame.nlargest(N, ['profit%'])

#Returns least profit stocks     
def most_loss(frame, N)
    return frame.nsmallest(N, ['profit%'])
       