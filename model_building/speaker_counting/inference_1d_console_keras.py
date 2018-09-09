import os
import datetime
import random as rng
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix

# Inputs
x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_test_clean/test-clean-features_40s_4c/x_test_normalized.txt'
y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_test_clean/test-clean-features_40s_4c/y_test.txt'
m_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/checkpoints/100ms_fft_env_hist/model2/the_network.h5'

def main(_):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ###########################################################################
    # Load model
    ###########################################################################

    the_network = keras.models.load_model(m_filename)
    print(the_network.summary())
    input("Press Enter to continue...")

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    t_start = time.time()

    # Load Input Files
    x_input     = np.loadtxt(x_filename, dtype=np.float32)
    y_input     = np.loadtxt(y_filename, dtype=np.float32)
    sz_set      = x_input.shape[0]
    sz_input    = x_input.shape[1]
    n_classes   = y_input.shape[1]

    print(type(x_input[0][0]))
    print(type(y_input[0][0]))

    t_stop = time.time()

    # Debug Messages
    print("Input prepare time   : ", str(t_stop - t_start))
    print("Total inputs         : ", str(sz_set))
    print("Input length         : ", str(sz_input))
    print("Number of classes    : ", str(n_classes))

    ###########################################################################
    # Run Inference
    ###########################################################################
    
    # Prepare input shape
    x_input = np.expand_dims(x_input, axis = 2)

    # Do inference to obtain confusion matrix
    t_start = time.time()
    y_prob  = the_network.predict(x_input)
    y_pred  = np.argmax(y_prob, axis=1)
    y_true  = np.argmax(y_input, axis=1)
    t_stop  = time.time()

    print("Inference time       : ", str(t_stop - t_start))
    print("Confusion matrix     :")
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':

    tf.app.run()
