import os
import random as rng
import time
import numpy as np
import tensorflow as tf

# Inputs
x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/dev-clean-features_30s_4c/x_train_normalized.txt'
y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/librispeech_dev_clean/dev-clean-features_30s_4c/y_train.txt'
s_model_save_dir = 'E:/1_Proiecte_Curente/1_Speaker_Counting/checkpoints/'

# x_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/x_dummy.txt'
# y_filename = 'E:/1_Proiecte_Curente/1_Speaker_Counting/datasets/y_dummy.txt'
# s_model_save_dir = 'E:/1_Proiecte_Curente/1_Speaker_Counting/checkpoints/'

# Architecture
n_first_layer_multiplier = 1.5
n_convolutional_layers = 4
n_dense_layers = 8
n_filt_pl = 60
n_filt_sz = 8
do_bn_conv = 1
do_bn_dense = 1
do_dropout_dense = 1
n_drop_jump = 3
n_bn_jump = 4

# Convergence
f_start_lr = 0.01
f_momentum = 0.98
f_decay_rate = 0.96
n_lr_gstep = 1000
n_lr_dstep = 500

# Training
f_use_for_validation = 0.005
f_use_for_inference = 0.001
sz_batch = 64
f_bn_epsilon = 1e-9
f_dropout_prob = 0.85
n_max_iterations = 30000
n_iterations_for_stop = 5000
n_iterations_for_sleep = 1000
n_sleep_seconds = 60

# Plotting & debugging
b_print_output = 1
n_print_interval = 200


def print_confusion_matrix(v_predicted, v_true, n_classes):
    m_confusion = np.zeros([n_classes, n_classes], dtype=int)
    n_samples = v_predicted.shape[0]

    for i in range(n_samples):
        i_pred = v_predicted[i]
        i_true = v_true[i]
        m_confusion[i_true][i_pred] += 1

    print("")
    print("Confusion Matrix: ")
    print(m_confusion)
    print("")

    return [m_confusion]


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
    sz_inference = int(sz_set * f_use_for_inference)
    sz_train = int(sz_set - sz_validate - sz_inference)
    sz_input = inputs.shape[1]
    n_batches = int(sz_train / sz_batch)

    t_stop = time.time()

    # Debug Messages
    print("Input prepare time   : ", str(t_stop - t_start))
    print("Total inputs         : ", str(sz_set))
    print("Input length         : ", str(sz_input))
    print("Number of classes    : ", str(n_classes))
    print("Used for training    : ", str(sz_train))
    print("Used for inference   : ", str(sz_inference))
    print("Used for validation  : ", str(sz_validate))
    print("Batch size           : ", str(sz_batch))

    ###########################################################################
    # Split Data into Train, Test and Validate
    ###########################################################################

    x_train = inputs[0:sz_train][:]
    x_inference = inputs[sz_train:sz_train+sz_inference][:]
    x_validate = inputs[sz_train+sz_inference:sz_set][:]
    y_train = outputs[0:sz_train][:]
    y_inference = outputs[sz_train:sz_train+sz_inference][:]
    y_validate = outputs[sz_train+sz_inference:sz_set][:]

    ###########################################################################
    # Build Model Graph
    ###########################################################################

    # Data Layer
    x_ = tf.placeholder(tf.float32, [None, sz_input])

    ###########################################################################
    # Targeted Neural Network Architecture
    ###########################################################################

    v_activations = []
    v_filters = []
    v_biases = []

    # First layer of the network

    sz_input_ann = sz_input
    x_final_conv = x_
    idx_last = -1

    if n_convolutional_layers > 0:

        idx_last = 0

        # First Convolutional Layer

        sz_input_decrease = n_filt_sz - 1
        xc_t = tf.reshape(x_, [-1, sz_input, 1])
        wc_0 = tf.Variable(tf.random_normal([n_filt_sz, 1, n_filt_pl], mean=0.0), dtype=tf.float32)
        bc_0 = tf.Variable(tf.zeros([sz_input - sz_input_decrease, n_filt_pl]), dtype=tf.float32)
        xc_0 = tf.sigmoid(tf.nn.conv1d(xc_t, wc_0, stride=1, padding='VALID') + bc_0)

        v_filters.append(wc_0)
        v_biases.append(bc_0)
        v_activations.append(xc_0)

        # Remaining Convolutional Layers

        for i in range(1, n_convolutional_layers):
            sz_input_new = sz_input - (i + 1) * sz_input_decrease
            wc_i = tf.Variable(tf.random_normal([n_filt_sz, n_filt_pl, n_filt_pl], mean=0.0), dtype=tf.float32)
            bc_i = tf.Variable(tf.zeros([sz_input_new, n_filt_pl]), dtype=tf.float32)
            xc_i = tf.sigmoid(tf.nn.conv1d(v_activations[idx_last], wc_i, stride=1, padding='VALID') + bc_i)           

            v_filters.append(wc_i)
            v_biases.append(bc_i)
            v_activations.append(xc_i)
            idx_last += 1

        sz_input_new = sz_input - n_convolutional_layers * sz_input_decrease
        sz_input_ann = sz_input_new * n_filt_pl
        x_final_conv = tf.reshape(v_activations[idx_last], [-1, sz_input_ann])

        # Do batch normalization
        if (do_bn_conv == 1):
            bn_conv_mean, bn_conv_var = tf.nn.moments(x_final_conv, [0])
            x_final_conv = tf.nn.batch_normalization(x_final_conv, bn_conv_mean, bn_conv_var, None, None, f_bn_epsilon)

    # First Densely Connected Layer

    sz_layer = int(sz_input * n_first_layer_multiplier)
    wd_0 = tf.Variable(tf.random_normal([sz_input_ann, sz_layer], mean=0.0), dtype=tf.float32)
    bd_0 = tf.Variable(tf.zeros([sz_layer]), dtype=tf.float32)
    xd_0 = tf.sigmoid(tf.matmul(x_final_conv, wd_0) + bd_0)
    
    v_filters.append(wd_0)
    v_biases.append(bd_0)
    v_activations.append(xd_0)
    idx_last += 1

    # Remaining Densely Connected Layers

    for i in range(1, n_dense_layers):
        wd_i = tf.Variable(tf.random_normal([sz_layer, sz_layer], mean=0.0), dtype=tf.float32)
        bd_i = tf.Variable(tf.zeros([sz_layer]), dtype=tf.float32)
        xd_i = tf.matmul(v_activations[idx_last], wd_i)

        if (do_bn_dense == 1):
            if (i % n_bn_jump == 0):
                bn_dense_mean, bn_dense_var = tf.nn.moments(xd_i, [0])
                xd_i = tf.nn.batch_normalization(xd_i, bn_dense_mean, bn_dense_var, None, None, f_bn_epsilon)
                
        if (do_dropout_dense == 1):
            if (i % n_drop_jump == 0):
                xd_i = tf.nn.dropout(xd_i, f_dropout_prob)

        xd_i = tf.sigmoid(xd_i + bd_i)

        v_filters.append(wd_i)
        v_biases.append(bd_i)
        v_activations.append(xd_i)
        idx_last += 1

    # Final Layer

    w_final = tf.Variable(tf.random_normal([sz_layer, n_classes], mean=0.0), dtype=tf.float32)
    b_final = tf.Variable(tf.zeros([n_classes]))
    y = tf.matmul(v_activations[idx_last], w_final) + b_final

    ###########################################################################
    # Run Training
    ###########################################################################

    # Create Training Method
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Create Gradient Descent Optimizer
    f_learning_rate = tf.train.exponential_decay(f_start_lr, n_lr_gstep, n_lr_dstep, f_decay_rate, staircase=True)
    train_step = tf.train.MomentumOptimizer(f_learning_rate,
                                            f_momentum,
                                            use_locking=False,
                                            use_nesterov=True).minimize(cost_function)

    # Create Validation / Inference Method
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_predicted_tensor = tf.argmax(y, 1)

    # Create Session
    sess = tf.InteractiveSession()

    # Create Variable Saver
    model_saver = tf.train.Saver()

    # Run Training
    tf.initialize_all_variables().run()

    t_start = time.time()

    # Used for automated training
    n_iteration_current = 0
    n_iterations_since_sleep = 0
    n_iterations_since_print = 0
    f_max_precision = 0.0

    # Used for real time plotting
    x_iteration_count = list()
    y_train_error = list()
    y_validate_error = list()

    b_stop = 0
    while b_stop == 0:

        # Use a mechanism to let the CPU and GPU cool when doing long training
        n_iterations_since_sleep += 1
        if n_iterations_since_sleep == n_iterations_for_sleep:
            time.sleep(n_sleep_seconds)  # Sleep for a little to let CPU and GPU cool
            n_iterations_since_sleep = 0

        # Get a random batch
        n_batch = rng.randint(0, n_batches - 1)
        n_iteration_current += 1

        if n_iteration_current == n_max_iterations:
            b_stop = 1

        batch_xs = x_train[(n_batch * sz_batch):((n_batch + 1) * sz_batch)][:]
        batch_ys = y_train[(n_batch * sz_batch):((n_batch + 1) * sz_batch)][:]
        sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})

        # Build Confusion Matrix
        f_train_acc = sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys})
        f_acc = sess.run(accuracy, feed_dict={x_: x_validate, y_: y_validate})
        y_predicted = sess.run(y_predicted_tensor, feed_dict={x_: x_validate, y_: y_validate})

        x_iteration_count.append(n_iteration_current)
        y_train_error.append(1 - f_train_acc)
        y_validate_error.append(1 - f_acc)

        # Display output in console
        if b_print_output:
            print("Iter.: %d; Validation: %.8f Training: %.8f" % (n_iteration_current, f_acc, f_train_acc))
            n_iterations_since_print += 1
            if n_iterations_since_print == n_print_interval:
                # Print confusion matrix
                y_true = np.argmax(y_validate, 1)
                print_confusion_matrix(y_predicted, y_true, n_classes)
                n_iterations_since_print = 0

        f_precision = f_acc

        # Stop training when the error is not decreasing for a certain number of iterations.
        if f_precision > f_max_precision:
            f_max_precision = f_precision

            # Save the model
            s_path = s_model_save_dir + str(f_precision) + "_" + str(n_iteration_current) + ".ckpt"
            model_save_path = model_saver.save(sess=sess, save_path=s_path)
            print("Model saved in file: %s" % model_save_path)

            # Print confusion matrix
            y_true = np.argmax(y_validate, 1)
            print_confusion_matrix(y_predicted, y_true, n_classes)

    t_stop = time.time()
    print("Training time        : " + str(t_stop - t_start))

    ###########################################################################
    # Run Inference
    ###########################################################################

    f_model_accuracy = sess.run(accuracy, feed_dict={x_: x_inference, y_: y_inference})
    print("Inference accuracy   : " + str(f_model_accuracy))


if __name__ == '__main__':

    tf.app.run()
