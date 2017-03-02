import time
import numpy as np
import tensorflow as tf
import gc
# import os

x_filename = './x_external.txt'
y_filename = './y_external.txt'
m_filename = './ann_model.ckpt'

def main(_):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    t_start = time.time()

    # Load Input Files
    inputs_t = np.loadtxt(x_filename)

    # Experiment Parameters
    n_classes = 1  # outputs_t.shape[1]
    sz_set = inputs_t.shape[0]
    sz_input = inputs_t.shape[1]

    # Trick python into knowing the size of _y tensor
    inputs = np.zeros([sz_set, sz_input], dtype=float, order='C')
    for i in range(sz_set):
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
    # Reconstruct the Model
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
    y_decision = tf.to_float(y > 0.8)

    ###########################################################################
    # Restore Model
    ###########################################################################

    # Create Session
    sess = tf.InteractiveSession()

    # Create Variable Saver
    model_saver = tf.train.Saver()

    # Load Model
    model_saver.restore(sess, save_path=m_filename)

    ###########################################################################
    # Run Forward Pass
    ###########################################################################

    n_batch_size = 5000
    n_current_idx = 1
    y_output = [[0.]]

    while (n_current_idx + n_batch_size - 1) < sz_set:
        batch_xs = inputs[n_current_idx:(n_current_idx + n_batch_size - 1)][:]
        v_output = sess.run(y_decision, feed_dict={x_: batch_xs})
        y_output = tf.concat(0, [y_output, v_output])

        n_current_idx += n_batch_size

        print("Forward pass one batch ...")

    batch_xs = inputs[n_current_idx - n_batch_size:sz_set][:]
    v_output = sess.run(y_decision, feed_dict={x_: batch_xs})
    y_output = tf.concat(0, [y_output, v_output])

    # Save Output
    y_output = y_output.eval()[2:sz_set + 2][:]
    np.savetxt(y_filename, y_output)

if __name__ == '__main__':

    tf.app.run()
