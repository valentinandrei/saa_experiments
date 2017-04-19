import time
import numpy as np
import tensorflow as tf
import gc
# import os

x_filename = '/home/valentin/Working/saa_experiments_db/valentin_ann_features/ms_500/x_test_normalized.txt'
y_filename = '/home/valentin/Working/saa_experiments_db/valentin_ann_features/ms_500/y_test.txt'
#m_filename = '/home/valentin/Working/saa_experiments/model_building/milestones/1_overlap_detection_500ms/model_500ms_4c_6d.ckpt'
m_filename = './0.8536_5946.ckpt'

# Architecture
n_first_layer_multiplier = 1.5
n_convolutional_layers = 3
n_dense_layers = 7
n_filt_pl = 10
n_filt_sz = 7


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
    n_classes = 1
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

    v_activations = []
    v_filters = []
    v_biases = []
    idx_last = 0

    if n_convolutional_layers > 0:

        # First Convolutional Layer

        sz_input_decrease = n_filt_sz - 1
        xc_t = tf.reshape(x_, [-1, sz_input, 1])
        wc_0 = tf.Variable(tf.random_normal([n_filt_sz, 1, n_filt_pl], mean=0.0), dtype=tf.float32)
        bc_0 = tf.Variable(tf.random_normal([sz_input - sz_input_decrease, n_filt_pl]), dtype=tf.float32)
        xc_0 = tf.sigmoid(tf.nn.conv1d(xc_t, wc_0, stride=1, padding='VALID') + bc_0)

        v_filters.append(wc_0)
        v_biases.append(bc_0)
        v_activations.append(xc_0)

        # Remaining Convolutional Layers

        for i in range(1, n_convolutional_layers):
            sz_input_new = sz_input - (i + 1) * sz_input_decrease
            wc_i = tf.Variable(tf.random_normal([n_filt_sz, n_filt_pl, n_filt_pl], mean=0.0), dtype=tf.float32)
            bc_i = tf.Variable(tf.random_normal([sz_input_new, n_filt_pl]), dtype=tf.float32)
            xc_i = tf.sigmoid(tf.nn.conv1d(v_activations[idx_last], wc_i, stride=1, padding='VALID') + bc_i)

            v_filters.append(wc_i)
            v_biases.append(bc_i)
            v_activations.append(xc_i)
            idx_last += 1

        sz_input_new = sz_input - n_convolutional_layers * sz_input_decrease
        sz_input_ann = sz_input_new * n_filt_pl
        x_final_conv = tf.reshape(v_activations[idx_last], [-1, sz_input_ann])

    else:

        sz_input_ann = sz_input
        x_final_conv = x_
        idx_last = -1

    # First Densely Connected Layer

    sz_layer = int(sz_input * n_first_layer_multiplier)
    wd_0 = tf.Variable(tf.random_normal([sz_input_ann, sz_layer], mean=0.0), dtype=tf.float32)
    bd_0 = tf.Variable(tf.random_normal([sz_layer]), dtype=tf.float32)
    xd_0 = tf.sigmoid(tf.matmul(x_final_conv, wd_0) + bd_0)

    v_filters.append(wd_0)
    v_biases.append(bd_0)
    v_activations.append(xd_0)
    idx_last += 1

    # Remaining Densely Connected Layers

    for i in range(1, n_dense_layers):
        wd_i = tf.Variable(tf.random_normal([sz_layer, sz_layer], mean=0.0), dtype=tf.float32)
        bd_i = tf.Variable(tf.random_normal([sz_layer]), dtype=tf.float32)
        xd_i = tf.sigmoid(tf.matmul(v_activations[idx_last], wd_i) + bd_i)

        v_filters.append(wd_i)
        v_biases.append(bd_i)
        v_activations.append(xd_i)
        idx_last += 1

    # Final Layer

    w_final = tf.Variable(tf.random_normal([sz_layer, n_classes], mean=0.0), dtype=tf.float32)
    b_final = tf.Variable(tf.random_normal([n_classes]))
    y = tf.sigmoid(tf.matmul(v_activations[idx_last], w_final) + b_final)

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
    # Run Inference
    ###########################################################################

    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Compute Per Class Scores
    t_pos = tf.logical_and(tf.cast(tf.round(y), tf.bool), tf.cast(y_, tf.bool))
    f_pos = tf.logical_and(tf.cast(tf.round(y), tf.bool), tf.logical_not(tf.cast(y_, tf.bool)))
    t_neg = tf.logical_and(tf.logical_not(tf.cast(tf.round(y), tf.bool)), tf.logical_not(tf.cast(y_, tf.bool)))
    f_neg = tf.logical_and(tf.logical_not(tf.cast(tf.round(y), tf.bool)), tf.cast(y_, tf.bool))

    r_t_pos = tf.reduce_mean(tf.cast(t_pos, tf.float32))
    r_f_pos = tf.reduce_mean(tf.cast(f_pos, tf.float32))
    r_t_neg = tf.reduce_mean(tf.cast(t_neg, tf.float32))
    r_f_neg = tf.reduce_mean(tf.cast(f_neg, tf.float32))

    f_model_accuracy = sess.run(accuracy, feed_dict={x_: inputs, y_: outputs})
    print("Inference accuracy   : " + str(f_model_accuracy))

    # Compute F-Score
    f_t_pos = sess.run(r_t_pos, feed_dict={x_: inputs, y_: outputs})
    f_f_pos = sess.run(r_f_pos, feed_dict={x_: inputs, y_: outputs})
    f_t_neg = sess.run(r_t_neg, feed_dict={x_: inputs, y_: outputs})
    f_f_neg = sess.run(r_f_neg, feed_dict={x_: inputs, y_: outputs})

    print("True positives       : " + str(f_t_pos))
    print("False positives      : " + str(f_f_pos))
    print("True negatives       : " + str(f_t_neg))
    print("False negatives      : " + str(f_f_neg))

    f_prec = f_t_pos / (f_t_pos + f_f_pos)
    f_recall = f_t_pos / (f_t_pos + f_f_neg)
    f_score = 2 * f_prec * f_recall / (f_prec + f_recall)

    print("Precision            : " + str(f_prec))
    print("Recall               : " + str(f_recall))
    print("F1-Score             : " + str(f_score))

if __name__ == '__main__':

    tf.app.run()
