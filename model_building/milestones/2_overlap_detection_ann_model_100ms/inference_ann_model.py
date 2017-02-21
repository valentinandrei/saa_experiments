import time
import numpy as np
import tensorflow as tf
import gc
# import os

x_filename = '/home/valentin/Working/phd_project/1_milestones/mfcc_10dl_model/x_test_normalized.txt'
y_filename = '/home/valentin/Working/phd_project/1_milestones/mfcc_10dl_model/y_test.txt'
m_filemane = '/home/valentin/Working/phd_project/1_milestones/mfcc_10dl_model/10dl_model.ckpt'

def main(_):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    t_start = time.time()

    # Load Input Files
    inputs_t = np.loadtxt(x_filename)
    outputs_t = np.loadtxt(y_filename)

    # Experiment Parameters
    n_classes = 1  # outputs_t.shape[1]
    sz_set = inputs_t.shape[0]
    sz_input = inputs_t.shape[1]

    # Trick python into knowing the size of _y tensor
    inputs = np.zeros([sz_set, sz_input], dtype=float, order='C')
    outputs = np.zeros([sz_set, n_classes], dtype=float, order='C')
    for i in range(sz_set):
        outputs[i][0] = outputs_t[i]
        for j in range(sz_input):
            inputs[i][j] = inputs_t[i][j]

    gc.collect()

    t_stop = time.time()
    print("Input prepare time   : ", str(t_stop - t_start))

    # Debug Messages
    print("Total inputs         : ", str(sz_set))
    print("Input length         : ", str(sz_input))
    print("Number of classes    : ", str(n_classes))

    ###########################################################################
    # Create Model
    ###########################################################################

    # Data Layer
    x_ = tf.placeholder(tf.float32, [None, sz_input])
    y_ = tf.placeholder(tf.float32, [None, 1])

    ###########################################################################
    # Targeted NN Architecture
    ###########################################################################

    n_dense_layers = 9
    n_layer_size = int(sz_input * 1.5)
    v_activations = []
    v_weights = []
    v_biases = []

    w_input = tf.Variable(tf.random_normal([sz_input, n_layer_size], mean=0.0), dtype=tf.float32)
    b_input = tf.Variable(tf.random_normal([n_layer_size]), dtype=tf.float32)
    x_input = tf.sigmoid(tf.matmul(x_, w_input) + b_input)

    v_activations.append(x_input)
    for i in range(1, n_dense_layers):
        w_temp = tf.Variable(tf.random_normal([n_layer_size, n_layer_size], mean=0.0), dtype=tf.float32)
        b_temp = tf.Variable(tf.random_normal([n_layer_size]), dtype=tf.float32)
        x_temp = tf.sigmoid(tf.matmul(v_activations[i - 1], w_temp) + b_temp)

        v_weights.append(w_temp)
        v_biases.append(b_temp)
        v_activations.append(x_temp)

    w_final = tf.Variable(tf.random_normal([n_layer_size, n_classes], mean=0.0), dtype=tf.float32)
    b_final = tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32)
    y = tf.sigmoid(tf.matmul(v_activations[n_dense_layers - 1], w_final) + b_final)

    ###########################################################################
    # Restore Model
    ###########################################################################

    # Create Session
    sess = tf.InteractiveSession()

    # Create Variable Saver
    model_saver = tf.train.Saver()

    # Load Model
    model_saver.restore(sess, save_path=m_filemane)

    ###########################################################################
    # Run Inference
    ###########################################################################

    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    f_model_accuracy = sess.run(accuracy, feed_dict={x_: inputs, y_: outputs})
    print("Inference accuracy   : " + str(f_model_accuracy))


if __name__ == '__main__':

    tf.app.run()
