import pickle

import pandas as pd
import numpy as np
import os
from model_utils import *
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select 0 for first GPU or 1 for second


def get_XY():
    df = pd.read_csv('./Trajectories.csv', sep=',', index_col=0)
    df_x = df.iloc[:, 0:-1]
    df_y = df.iloc[:, [-1]]
    # tmp = df_x.values.flatten()
    # tmp1 = np.isnan(tmp)
    # all_data = tmp[np.argwhere(tmp1 == False)]
    df_x = (df_x - 2.663817261047651) / 3.7609516508127996
    # np.mean(all_data), (np.std(all_data))

    df_x.fillna(0, inplace=True)

    x = df_x.values.reshape((df_x.shape[0], df_x.shape[1], 1))
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df_y['Traj_type'])
    class_num = np.unique(y).shape[0]
    # Congressing: 0, Quasi-static: 1, Retracing: 2
    y = to_categorical(y, num_classes=class_num)
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
    for train_index, test_index in sss.split(x, y):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]

    return x_train, y_train, x_test, y_test, class_num


cur_time = "test"
result_dir = './result'
model_to_save = result_dir + '/save_model_' + cur_time

x_train, y_train, x_test, y_test, class_num = get_XY()
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

model = get_TextRNN_model(maxlen=x_train.shape[1], class_num=class_num)
model.load_weights(model_to_save + ".h5")
best_loss = model.evaluate(x_test, y_test)

file_path = './Trajectories.training_data.csv'
df = pd.read_csv(file_path, sep=',', index_col=0)
df_x = df
df_x = (df_x - 2.663817261047651) / 3.7609516508127996
df_x.fillna(0, inplace=True)
x = df_x.values.reshape((df_x.shape[0], df_x.shape[1], 1))

y_pred = np.argmax(model.predict(x), axis=1)
labels = list(pd.Series(y_pred).map({0: 'Congressing', 1: 'Quasi-static', 2: 'Retracing'}))
df['label'] = labels
df.to_csv(file_path, sep=',')
