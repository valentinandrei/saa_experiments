import time
import numpy as np
import os
import tensorflow as tf
import random as rng

x_filename = '/home/valentin/Working/phd_project/dataset/x_spct_3_speakers_S1_9_200ms_100ms_inc_44kHz_5000.txt'
y_filename = '/home/valentin/Working/phd_project/dataset/y_spct_3_speakers_S1_9_200ms_100ms_inc_44kHz_5000.txt'

# x_filename = 'debug_x.txt'
# y_filename = 'debug_y.txt'


def gen_debug_data():

    t_start = time.time()

    n_train_samples = 20000
    n_input_sz = 100
    n_classes = 4

    n_total_samples = n_train_samples
    inputs = np.zeros([n_total_samples, n_input_sz], dtype=float, order='C')
    outputs = np.zeros([n_total_samples, n_classes], dtype=float, order='C')

    for i in range(n_total_samples):
        id_class = rng.randint(0, n_classes - 1)
        outputs[i][id_class] = 1

        for j in range(n_input_sz):
            inputs[i][j] = id_class + 1 / rng.randint(1, 64)

    t_stop = time.time()
    print("Data Generating Time (seconds): ", str(t_stop - t_start))

    return [inputs, outputs]


def main(_):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ###########################################################################
    # Generate Debug Data
    ###########################################################################

    # [inputs, outputs] = gen_debug_data()

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    t_start = time.time()

    # Load Input Files
    inputs = np.loadtxt(x_filename)
    outputs = np.loadtxt(y_filename)

    # Experiment Parameters
    n_classes = outputs.shape[1]
    n_batches = 75
    sz_set = inputs.shape[0]
    sz_validate = sz_set * 0.05
    sz_inference = sz_set * 0.05
    sz_train = sz_set - sz_validate - sz_inference
    sz_batch = np.round(sz_train / n_batches)
    sz_input = inputs.shape[1]

    t_stop = time.time()
    print("Input prepare time   : ", str(t_stop - t_start))

    # Debug Messages
    print("Total inputs         : ", str(sz_set))
    print("Input length         : ", str(sz_input))
    print("Number of classes    : ", str(n_classes))
    print("Used for training    : ", str(sz_train))
    print("Used for inference   : ", str(sz_inference))
    print("Used for validation  : ", str(sz_validate))
    print("Batch size           : ", str(sz_batch))

    # Split Data
    x_train = inputs[0:sz_train][:]
    x_inference = inputs[sz_train:sz_train+sz_inference][:]
    x_validate = inputs[sz_train+sz_inference:sz_set][:]
    y_train = outputs[0:sz_train][:]
    y_inference = outputs[sz_train:sz_train+sz_inference][:]
    y_validate = outputs[sz_train+sz_inference:sz_set][:]

    ###########################################################################
    # Build Model Graph
    ###########################################################################

    # Input Layer
    x_ = tf.placeholder(tf.float32, [None, sz_input])

    # Debug NN

    # w0 = tf.Variable(tf.zeros([sz_input, 1]), dtype=tf.float32)
    # b0 = tf.Variable(tf.zeros([n_classes]), dtype=tf.float32)
    # y = tf.matmul(x_, w0) + b0

    # Real NN

    # Neural Network Model Parameters
    n_c1_filter_order = 15
    n_c1_features = 15
    n_c2_filter_order = 10
    n_c2_features = 30
    n_c3_filter_order = 5
    n_c3_features = 5

    # Reshape
    x0_0 = tf.reshape(x_, shape=[-1, 1, sz_input, 1])

    # Convolutional layer 1
    w0 = tf.Variable(tf.truncated_normal([1, n_c1_filter_order, 1, n_c1_features]), dtype=tf.float32)
    x0_1 = tf.nn.conv2d(x0_0, w0, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
    x1 = x0_1

    # Convolutional layer 2
    w1 = tf.Variable(tf.truncated_normal([1, n_c2_filter_order, n_c1_features, n_c2_features]), dtype=tf.float32)
    x1_1 = tf.nn.conv2d(x1, w1, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
    x2 = x1_1

    # Convolutional layer 3
    w2 = tf.Variable(tf.truncated_normal([1, n_c3_filter_order, n_c2_features, n_c3_features]), dtype=tf.float32)
    x2_1 = tf.nn.conv2d(x2, w2, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
    x2_2 = tf.reshape(x2_1, [-1, sz_input * n_c3_features])
    x3 = x2_2

    # Fully connected layer
    w3 = tf.Variable(tf.zeros([sz_input * n_c3_features, n_classes]), dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([n_classes]), dtype=tf.float32)
    y = tf.sigmoid(tf.matmul(x3, w3) + b3)

    ###########################################################################
    # Run Training
    ###########################################################################

    # Training parameters
    f_learning_rate = 0.001

    # Create Training Method
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(f_learning_rate).minimize(cross_entropy)

    # Create Validation Method
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create Session
    sess = tf.InteractiveSession()

    # Run Training
    tf.initialize_all_variables().run()

    t_start = time.time()

    for i in range(n_batches):
        print("Batch                : " + str(i))
        batch_xs = x_train[(i * sz_batch):((i + 1) * sz_batch)][:]
        batch_ys = y_train[(i * sz_batch):((i + 1) * sz_batch)][:]
        sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})

        f_validation_accuracy = sess.run(accuracy, feed_dict={x_: x_validate, y_: y_validate})
        print("Validation accuracy  : " + str(f_validation_accuracy))

        if f_validation_accuracy > 0.8:
            break

    t_stop = time.time()
    print("Training time        : " + str(t_stop - t_start))

    ###########################################################################
    # Run Inference
    ###########################################################################

    # Compute Neural Network Accuracy
    f_model_accuracy = sess.run(accuracy, feed_dict={x_: x_inference, y_: y_inference})
    print("Model accuracy       : " + str(f_model_accuracy))


if __name__ == '__main__':

    tf.app.run()
