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


def plot_learning_curve(history, result_dir, k=None):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.plot(loss, 'b', label='loss')
    ax.plot(val_loss, 'r', label='val_loss')
    ax.legend(['loss', 'val_loss'])
    if k != None:
        plt.savefig(result_dir + '/fig' + '_' + str(k) + '.png', dpi=200)
    else:
        plt.savefig(result_dir + '/fig' + '.png', dpi=200)


batchsz = 60
epochs = 400

cur_time = "test"
result_dir = './result'
model_to_save = result_dir + '/save_model_' + cur_time
path_txt = model_to_save + '/logs_' + cur_time + '.txt'
if os.path.exists(path_txt):
    os.remove(path_txt)

x_train, y_train, x_test, y_test, class_num = get_XY()
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

model = get_TextRNN_model(maxlen=x_train.shape[1], class_num=class_num)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_to_save + ".h5",
        monitor='val_loss',
        save_best_only=True,
    ),
    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0001,
    #     patience=20,
    #     verbose=0,
    #     mode='auto',
    #     # baseline=90,
    #     # restore_best_weights=False
    # ),
]
#
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batchsz, callbacks=callbacks_list,
                    validation_data=(x_test, y_test), shuffle=False)

f_his = open(result_dir + '/history_' + cur_time + '.pickle', "wb")
pickle.dump(history.history, f_his)
f_his.close()
plot_learning_curve(history, result_dir=result_dir)

model_evaluate = get_TextRNN_model(maxlen=x_train.shape[1], class_num=class_num)
model_evaluate.load_weights(model_to_save + ".h5")
best_loss = model_evaluate.evaluate(x_test, y_test)
with open(result_dir + '/loss_valacc_' + cur_time + '.csv', 'a+') as f:
    f.write(str(best_loss) + '\n')

