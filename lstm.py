import gc
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.preprocessing import StandardScaler
from utils import get_LSTM_input
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

def MAPE(true,pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)*100

def get_train_and_test(ftrain,ftest,type_list):

    x_train, y_train, train_scalers = get_LSTM_input(ftrain)

    if type_list[0] == 'speed':
        y_train = y_train[:,0]
    if type_list[0] == 'flow':
        y_train = y_train[:,1]

    x_test, y_test, test_scalers = get_LSTM_input(ftest,True)

    x_test = np.array(list(x_test.values()))
    y_test = np.array(list(y_test.values()))

    if type_list[0] == 'speed':
        y_test = y_test[:,0]
    if type_list[0] == 'flow':
        y_test = y_test[:,1]

    print (x_train.shape)
    print (x_test.shape)
    print (y_train.shape)
    print (y_test.shape)

    return x_train,y_train,train_scalers,x_test,y_test,test_scalers

def train_lstm(x_train,y_train,type_list):
    model = Sequential()
    # model.add(LSTM(units = 64, input_shape = (100,3), return_sequences = True))
    model.add(LSTM(units=32, input_shape = (100,2), return_sequences = True))
    if (type_list[0]=='speed'):
        model.add(LSTM(units=32, input_shape = (100,2), return_sequences = True))
    if (type_list[0]=='flow'):
        model.add(LSTM(units=32, input_shape = (100,2), return_sequences = True))
        model.add(Dropout(0.3))

    model.add(LSTM(32, return_sequences=False)) # SET HERE
    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.summary()

    model.fit(x_train, y_train, epochs = 200, batch_size = 32)

    pickle_name = type_list[0]
    with open('pickle/lstm_'+ pickle_name +'.pickle','wb') as f:
        pickle.dump(model,f)


def test_lstm(x_test,y_test,scaler_speed,scaler_flow,type_list):


    pickle_name = type_list[0]

    with open('pickle/lstm_' + pickle_name + '.pickle','rb') as f:
        model = pickle.load(f)

    if type_list[0] == 'speed':
        pred=[]
        for i in x_test:
            speed = model.predict(i.reshape(1,100,2))
            pred.append(scaler_speed.inverse_transform(speed.tolist()).tolist()[0][0])

        # print (pred)
        x = list(range(len(y_test)))
        y_true = [scaler_speed.inverse_transform(i.reshape(-1,1)).tolist()[0][0] for i in y_test]

        print ('mae: ',mean_absolute_error(y_true,pred))
        print ('r2 score: ',r2_score(y_true,pred))
        print ('mse: ',mean_squared_error(y_true,pred))
        print ('mape: ',MAPE(y_true,pred))

        plt.rcParams['font.family'] = ['Ping Hei']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure()
        plt.plot(x,y_true,color = '#4285F4',label="真实值",linestyle='-')
        plt.plot(x,pred,color = '#DB4437',label= "预测值",linestyle='--',marker='x')
        plt.legend(loc='best',framealpha=0.5)
        plt.title('速度长短期记忆网络预测结果')
        plt.ylabel('速度')
        plt.show()

    if type_list[0] == 'flow':
        pred=[]
        for i in x_test:
            # flow = model.predict(i.reshape(1,100,3))
            flow = model.predict(i.reshape(1,100,2))
            pred.append(scaler_flow.inverse_transform(flow.tolist()).tolist()[0][0])

        # print (pred)
        x = list(range(len(y_test)))
        y_true = [scaler_flow.inverse_transform(i.reshape(-1,1)).tolist()[0][0] for i in y_test]

        print ('mae: ',mean_absolute_error(y_true,pred))
        print ('r2 score: ',r2_score(y_true,pred))
        print ('mse: ',mean_squared_error(y_true,pred))
        print ('mape: ',MAPE(y_true,pred))

        fig = plt.figure()
        plt.plot(x,pred,linestyle = ':',color = 'r',label = 'flow_predict')
        plt.plot(x,y_true,linestyle = '-',color = 'g',label = 'flow_true')
        plt.legend(loc='best',framealpha=0.5)
        plt.title('LSTM prediction for flow')
        plt.ylabel('flow')
        plt.show()






if __name__ == '__main__':
    type_list = ['speed']

    x_train,y_train,train_scalers,x_test,y_test,test_scalers  = get_train_and_test('train','test',type_list)
    train_lstm(x_train,y_train,type_list)

    scaler_speed, scaler_flow = test_scalers['speed'],test_scalers['flow']
    test_lstm(x_test,y_test,scaler_speed, scaler_flow,type_list)
