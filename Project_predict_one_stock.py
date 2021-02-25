#simple user interface. Gives predection of tomorrow's price of 
#a given stock (from 6365 stocks that have been trained)


import pandas_datareader as web
import pandas as pd
import numpy as np
from glob import glob
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
from PIL import Image 

#Predicting future prices for selected stock
def predict_tmr(web_stock_name, end_date):
    
    try:
        df = web.DataReader(web_stock_name, data_source='yahoo',start='2020-7-10', end=end_date)
    except:
        print('*Data not available  OR  wrong stock name!*')
        return
    new_df = df.filter(['Close'])
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(new_df)
    last_60_days = new_df[-60:].values
    print('Stock price of today: ' + str(last_60_days[-1]) + '\n')
    last_60_days_scaled = scaler.transform(last_60_days)  
    
    model = keras.models.load_model('C:/Users/tahaa/Desktop/python/AI_Project/Models/' + web_stock_name + '.us_model')
    
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    return (pred[0][0])




stock = input('Enter company name (use ticker symbol): ')
today = str(date.today())

#stock_list = []
#print(stock_list)
  
#Shows the graph of how well the model of this stock has been trained.
img = Image.open('C:/Users/tahaa/Desktop/python/AI_Project/Graphs/' + stock + '.us_graph.png')  
img.show() 

print('\nPredicted price for tomorrow: ' + str(predict_tmr(stock, today)))

