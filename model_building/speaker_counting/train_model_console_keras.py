import os
import datetime
import random as rng
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

# Inputs
x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/dev-clean-features_30s_4c/x_train_normalized.txt'
y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/dev-clean-features_30s_4c/y_train.txt'
s_model_save_dir = 'E:/1_Proiecte_Curente/1_Speaker_Counting/checkpoints/'

# x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/x_dummy.txt'
# y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/y_dummy.txt'
# s_model_save_dir = 'E:/1_Proiecte_Curente/1_Speaker_Counting/checkpoints/'

# Architecture
n_filters_L1        = 20
n_filters_L2        = 30
n_filters_L3        = 40
n_kernel_sz_L1      = 30
n_kernel_sz_L2      = 20
n_kernel_sz_L3      = 10
n_strides_L1        = 1
n_strides_L2        = 2
n_strides_L3        = 4
n_units_dense_L1    = 3072
n_units_dense_L2    = 2048
n_units_dense_L3    = 1024
f_dropout_prob_L1   = 0.5
f_dropout_prob_L2   = 0.5
f_dropout_prob_L3   = 0.5

# Training
f_use_for_validation = 0.005
sz_batch = 64
n_epochs = 40
f_start_lr = 0.0001

# Plotting & debugging
# TODO

def main(_):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    t_start = time.time()

    # Load Input Files
    inputs = np.loadtxt(x_filename)
    outputs = np.loadtxt(y_filename)

    # Experiment Parameters
    n_classes = outputs.shape[1]
    sz_set = inputs.shape[0]
    sz_validate = int(sz_set * f_use_for_validation)
    sz_train = int(sz_set - sz_validate)
    sz_input = inputs.shape[1]

    t_stop = time.time()

    # Debug Messages
    print("Input prepare time   : ", str(t_stop - t_start))
    print("Total inputs         : ", str(sz_set))
    print("Input length         : ", str(sz_input))
    print("Number of classes    : ", str(n_classes))
    print("Used for training    : ", str(sz_train))
    print("Used for validation  : ", str(sz_validate))

    ###########################################################################
    # Split Data into Train and Validate
    ###########################################################################

    x_train = inputs[0:sz_train][:]
    x_validate = inputs[sz_train:sz_set][:]
    y_train = outputs[0:sz_train][:]
    y_validate = outputs[sz_train:sz_set][:]

    # Expanding dimensions to be able to use Conv 1D

    x_train = np.expand_dims(x_train, axis = 2)
    x_validate = np.expand_dims(x_validate, axis = 2)

    ###########################################################################
    # Targeted Neural Network Architecture
    ###########################################################################

    the_network = keras.Sequential()

    the_network.add(keras.layers.Conv1D(filters = n_filters_L1, 
                                        kernel_size = (n_kernel_sz_L1), 
                                        strides = n_strides_L1))

    the_network.add(keras.layers.Conv1D(filters = n_filters_L2, 
                                        kernel_size = (n_kernel_sz_L2), 
                                        strides = n_strides_L2))

    the_network.add(keras.layers.Conv1D(filters = n_filters_L3, 
                                        kernel_size = (n_kernel_sz_L3), 
                                        strides = n_strides_L3))
    
    the_network.add(keras.layers.Dropout(f_dropout_prob_L1))

    the_network.add(keras.layers.BatchNormalization())

    the_network.add(keras.layers.Flatten())

    the_network.add(keras.layers.Dense(n_units_dense_L1, activation='sigmoid'))

    the_network.add(keras.layers.Dense(n_units_dense_L2, activation='relu'))

    the_network.add(keras.layers.Dropout(f_dropout_prob_L2))

    the_network.add(keras.layers.Dense(n_units_dense_L3, activation='relu'))

    the_network.add(keras.layers.Dense(n_classes, activation='softmax'))

    ###########################################################################
    # Run Training
    ###########################################################################

    the_network.compile(optimizer=tf.train.AdamOptimizer(f_start_lr), 
                        loss=keras.losses.categorical_crossentropy, 
                        metrics=['categorical_accuracy'])
   
    s_log_file = s_model_save_dir + "the_network_log.csv"

    csv_logger = CSVLogger(s_log_file, append=True, separator=';')
    
    model_saver = keras.callbacks.ModelCheckpoint(s_model_save_dir + "the_network.h5", 
                                                  monitor='val_categorical_accuracy', 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  save_weights_only=False, 
                                                  mode='auto', 
                                                  period=1)

    t_start = time.time()
    the_network.fit(x = x_train, 
                    y = y_train, 
                    epochs=n_epochs, 
                    batch_size=sz_batch, 
                    validation_data=(x_validate, y_validate),
                    callbacks=[csv_logger, model_saver])
    
    t_stop = time.time()
    print("Training time : " + str(t_stop - t_start))

if __name__ == '__main__':

    tf.app.run()
