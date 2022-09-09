import tensorflow as tf
import os
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import Lambda, Flatten, Dense, GlobalMaxPool1D, Activation, Conv1D, GRU, \
    GlobalMaxPooling1D, \
    Dropout, Average, LSTM, Layer, GlobalAveragePooling1D, BatchNormalization, Concatenate, MaxPool1D, Masking
from tensorflow.keras import initializers, Input, Model, constraints, layers, losses, optimizers, \
    Sequential, Input, regularizers
import copy
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select 0 for first GPU or 1 for second
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, text_path):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
        self.text_path = text_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = np.argmax(self.validation_data[1], axis=-1)

        # if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
        #     val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        with open(self.text_path, 'a+') as f:
            f.write(str(_val_f1.item()) + " " + str(_val_recall.item()) + " " + str(_val_precision.item()) + " ")
            f.write("\n")
        return


def get_TextRNN_model(maxlen, class_num, last_activation='softmax'):
    input = Input((maxlen, 1,))
    x = Masking(mask_value=0.0)(input)
    x = GRU(128, dropout=0.1, return_sequences=True)(x)
    x = GRU(64, dropout=0.1)(x)
    # x = GlobalMaxPool1D()(x)
    x = Sequential([
        layers.Dense(16, kernel_regularizer=regularizers.l2(0.1)),
        layers.Dropout(rate=0.1),
        layers.ReLU(),
        layers.Dense(8, kernel_regularizer=regularizers.l2(0.1)),
        layers.Dropout(rate=0.1),
        layers.ReLU()
    ])(x)

    output = Dense(class_num, activation=last_activation, kernel_regularizer=regularizers.l2(0.01))(x)
    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.0005,
                                            clipnorm=1.,
                                            decay=0.000)  # , experimental_run_tf_function=False
                  , loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
