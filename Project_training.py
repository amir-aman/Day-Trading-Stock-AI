#Training data and creating models for each stock in the database(size:6365)
#Used LSTM DNN algorithm with 2 hidden layer.

import math
import os
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from glob import glob
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')



#Reading all datafiles and storing them in one list (dfs)
data_files = glob('C:/Users/tahaa/Desktop/python/AI_Project/Stocks/*.txt')
dfs = [pd.read_csv(data_file) for data_file in data_files]

#for df in dfs:
for i in range(0, len(dfs)):

    #Returns stock name for graph title
    base = os.path.basename(data_files[i])
    os.path.splitext(base)
    stock_name = os.path.splitext(base)[0]
    web_stock_name = stock_name.replace('.us', '').upper()
    
    df = dfs[i]

    #Plot dataset
    df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index=df['Date']
    
#    plt.figure(figsize=(18,6))
#    plt.title(stock_name + ' Price History')
#    plt.plot(df['Close'])
#    plt.xlabel('Date', fontsize=14)
#    plt.ylabel('Close Price USD$', fontsize=14)
#    plt.show()

    #Create training data length
    data = df.filter(['Close'])
    dataset = data.values
    training_data_length = math.ceil(len(dataset)*0.8)

    #Scale the Data for easier training
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    #Create the scaled training dataset
    train_data = scaled_data[0:training_data_length, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    #Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)


    #Reshape data from 2D to 3D
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    #Creating test dataset
    test_data = scaled_data[training_data_length - 60:, :]
    
    #Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_length:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    #Convert data to numpy array
    x_test = np.array(x_test)


    #Reshape data from 2D to 3D
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Get predicted values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    #Get error using RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
    rmse = round(rmse, 4)
    total_rmse += rmse

    #Plot trained and tested data
    train = data[:training_data_length]
    valid = data[training_data_length:]
    valid['Predictions'] = predictions

    #Visualization    
    plt.figure(figsize=(18,6))
    plt.title('Prediction vs Actual Stock Price for ' + stock_name)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD$', fontsize=14)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
    plt.figtext(0, 0, 'RMSE = ' + str(rmse), style='italic', color='red', fontsize=12,
                bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    plt.savefig(stock_name + '_graph.png', bbox_inches='tight')
    plt.close()
#     plt.show()
    
    model.save(stock_name + '_model')
    
#    print(valid)
    
