import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler


def create_ar_time_series(n=1000,lag=1,phi=[0.8]):
    data = np.zeros(shape=n)
    for tau in np.arange(lag):
        data[tau] = np.random.normal(0,1,size=1)
    for i in np.arange(lag,n):
        data[i] = np.sum([phi[j-1]*data[j-1] for j in np.arange(1,lag+1)]) + np.random.normal(0,1,size=1) [0]
    return data

def main():
    LAG = 1
    N = 1000
    PHI = [0.8]
    data = create_ar_time_series(n=N,lag=LAG,phi=PHI)

    TIMESTEPS = 10
    BATCH_SIZE = 16


    train_length = int(data.shape[0]*0.8)
    train_data = data[:train_length]
    test_data = data[train_length+10:]

    scaler = MinMaxScaler(feature_range = (0,1))
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1,1))

    X_train = []
    y_train = []
    for i in np.arange(TIMESTEPS,train_length):
        X_train.append(train_data_scaled[i-TIMESTEPS:i])
        y_train.append(train_data_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    y_train = np.reshape(y_train,(-1,1))

    model = Sequential()
    model.add(LSTM(units=4,input_shape=(X_train.shape[1],1),return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=4,input_shape=(X_train.shape[1],1),return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=100)

    X_test_scaled = []
    y_test = []
    test_data_scaled = scaler.transform(test_data.reshape(-1,1))
    for j in np.arange(TIMESTEPS,test_data.shape[0]):
        X_test_scaled.append(test_data_scaled[j-TIMESTEPS:j])
        y_test.append(test_data[j])
    X_test_scaled, y_test = np.array(X_test_scaled), np.array(y_test)
    X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0],X_test_scaled.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    model.summary()

    # model.evaluate(X_test,y_test,batch_size=BATCH_SIZE)

    predictions = model.predict(X_test_scaled)
    predictions = scaler.inverse_transform(predictions)
    error = (predictions-y_test)
    rmse = np.sqrt(np.mean(error**2))
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(predictions)
    ax.plot(y_test)

    print(f'RMSE = {rmse}')
    print(1)

if __name__ == '__main__':
    main()