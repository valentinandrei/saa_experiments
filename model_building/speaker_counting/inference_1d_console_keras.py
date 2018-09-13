import os
import datetime
import random as rng
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix

# Inputs 100ms_specgram_env_hist_40s
x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_test_clean/100ms_specgram_env_hist_40s/x_test_normalized.txt'
y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_test_clean/100ms_specgram_env_hist_40s/y_test.txt'
m_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/checkpoints/100ms_specgram_env_hist/model2/the_network.h5'

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
    c_matrix = confusion_matrix(y_true, y_pred)
    print(c_matrix)

    f_avg_accuracy = 0.0
    for i in range(n_classes):
        n_sum = sum(c_matrix[i][:])
        f_accuracy = c_matrix[i][i] / n_sum
        f_avg_accuracy += f_accuracy
        print("Class {:d} accuracy: {:.2f}%".format(i + 1, 100.0 * f_accuracy))
    
    print("inference categorical accuracy: {:.2f}%".format(f_avg_accuracy / n_classes * 100.0))

    t_p = c_matrix[0][0]
    f_p = sum(c_matrix[0][1:])
    f_n = 0
    for i in range(n_classes - 1):
        f_n += c_matrix[i + 1][0]
    
    od_prec = t_p / (t_p + f_p)
    od_rec = t_p / (t_p + f_n)
    print("overlap detection precision: {:.2f}".format(od_prec))
    print("overlap detection recall: {:.2f}".format(od_rec))
    print("overlap detection F-Score: {:.2f}".format(2 * od_rec * od_prec / (od_rec + od_prec)))

if __name__ == '__main__':

    tf.app.run()