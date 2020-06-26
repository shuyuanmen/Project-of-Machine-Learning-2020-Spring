import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

def parsing_time(s):
    s = s[9:]
    i = 0
    for i in range(len(s)):
        if s[i] == ':':
            break
    a = int(s[:i])
    b = int(s[i + 1:])
    return a + b / 60.0

def load_data_speed():
    x_train = np.zeros([2016 - 1, 3])
    y_train = np.zeros(2016 - 1)
    x_test = np.zeros([576 - 1, 3])
    y_test = np.zeros(576 - 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    with open('train.csv', newline='', encoding='utf-8') as f_train:
        reader = csv.reader(f_train)
        i = 0
        for row in reader:
            if i < 2016 - 1:
                x_train[i, 0] = parsing_time(row[0])
                x_train[i, 1] = row[1]
                x_train[i, 2] = row[2]
            if i >= 1:
                y_train[i - 1] = row[1]
            i += 1
    with open('test.csv', newline='', encoding='utf-8') as f_test:
        reader = csv.reader(f_test)
        i = 0
        for row in reader:
            if i < 576 - 1:
                x_test[i, 0] = parsing_time(row[0])
                x_test[i, 1] = row[1]
                x_test[i, 2] = row[3]
            if i >= 1:
                y_test[i - 1] = row[1]
            i += 1
#    for i in range(3):
#        Max = max(x_train[:, i])
#        Min = min(x_train[:, i])
#        x_train[:, i] = (x_train[:, i] - Min) / (Max - Min)
#    Max = max(y_train[:])
#    Min = min(y_train[:])
#    y_train[:] = (y_train[:] - Min) / (Max - Min)
#    for i in range(3):
#        Max = max(x_test[:, i])
#        Min = min(x_test[:, i])
#        x_test[:, i] = (x_test[:, i] - Min) / (Max - Min)
#    Max = max(y_test[:])
#    Min = min(y_test[:])
#    y_test[:] = (y_test[:] - Min) / (Max - Min)
#    x_train = x_train / 24.0
#    y_train = y_train / 100.0
#    x_test = x_test / 24.0
#    y_test = y_test / 100.0
    return (x_train, y_train), (x_test, y_test)


(speed_x_train, speed_y_train), (speed_x_test, speed_y_test) = load_data_speed()
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)
model = Sequential()
model.add(Dense(input_dim=3, units=1000, activation='relu'))
for i in range(10):
    model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1, activation='relu'))

speed_train_Sum = 0.0
speed_test_Sum = 0.0
converge = 0
for i in range(10):
    model.compile(loss='mape', optimizer='adamax')
    model.fit(speed_x_train, speed_y_train, batch_size=100, epochs=100)
    train_score = model.evaluate(speed_x_train, speed_y_train)
    test_score = model.evaluate(speed_x_test, speed_y_test)
    if train_score < 100:
        speed_train_Sum += train_score
        speed_test_Sum += test_score
        converge += 1
print('The training error of speed is ', speed_train_Sum / converge)
print('The testing error of speed is ', speed_test_Sum / converge)




def load_data_flow():
    x_train = np.zeros([2016 - 1, 3])
    y_train = np.zeros(2016 - 1)
    x_test = np.zeros([576 - 1, 3])
    y_test = np.zeros(576 - 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    with open('train.csv', newline='', encoding='utf-8') as f_train:
        reader = csv.reader(f_train)
        i = 0
        for row in reader:
            if i < 2016 - 1:
                x_train[i, 0] = parsing_time(row[0])
                x_train[i, 1] = row[1]
                x_train[i, 2] = row[2]
            if i >= 1:
                y_train[i - 1] = row[2]
            i += 1
    with open('test.csv', newline='', encoding='utf-8') as f_test:
        reader = csv.reader(f_test)
        i = 0
        for row in reader:
            if i < 576 - 1:
                x_test[i, 0] = parsing_time(row[0])
                x_test[i, 1] = row[1]
                x_test[i, 2] = row[3]
            if i >= 1:
                y_test[i - 1] = row[3]
            i += 1
#    for i in range(3):
#        Max = max(x_train[:, i])
#        Min = min(x_train[:, i])
#        x_train[:, i] = (x_train[:, i] - Min) / (Max - Min)
#    Max = max(y_train[:])
#    Min = min(y_train[:])
#    y_train[:] = (y_train[:] - Min) / (Max - Min)
#    for i in range(3):
#        Max = max(x_test[:, i])
#        Min = min(x_test[:, i])
#        x_test[:, i] = (x_test[:, i] - Min) / (Max - Min)
#    Max = max(y_test[:])
#    Min = min(y_test[:])
#    y_test[:] = (y_test[:] - Min) / (Max - Min)
#    x_train = x_train / 24.0
#    y_train = y_train / 100.0
#    x_test = x_test / 24.0
#    y_test = y_test / 100.0
    return (x_train, y_train), (x_test, y_test)

(flow_x_train, flow_y_train), (flow_x_test, flow_y_test) = load_data_flow()
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)
model = Sequential()
model.add(Dense(input_dim=3, units=1024, activation='relu'))
for i in range(8):
    model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=1, activation='relu'))

flow_train_Sum = 0.0
flow_test_Sum = 0.0
converge = 0
for i in range(8):
    model.compile(loss='mape', optimizer='adamax')
    model.fit(flow_x_train, flow_y_train, batch_size=128, epochs=128)
    train_score = model.evaluate(flow_x_train, flow_y_train)
    test_score = model.evaluate(flow_x_test, flow_y_test)
    if train_score < 100:
        flow_train_Sum += train_score
        flow_test_Sum += test_score
        converge += 1
print('The training error of flow is ', flow_train_Sum / converge)
print('The testing error of flow is ', flow_test_Sum / converge)
