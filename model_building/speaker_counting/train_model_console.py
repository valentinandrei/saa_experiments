import time
import numpy as np
import tensorflow as tf
import random as rng
import gc

# Inputs
x_filename = '/home/valentin/Working/saa_experiments/data_building/octave_scripts/x_train_normalized.txt'
y_filename = '/home/valentin/Working/saa_experiments/data_building/octave_scripts/y_train.txt'
s_model_save_dir = '/home/valentin/Working/saa_experiments/model_building/speaker_counting/'

# Optimal parameters
# 1: LR 0.01, BS 400, Decay 0.98 -> 100% on debug data and altered real data
# 2: CL 10, FSZ 15 FPL 25 LR 0.0001 M 1.0 D 0.96 -> 100% on first inputs

# Dataset Splitting
n_batches = 200

# Architecture
n_first_layer_multiplier = 1.5
n_convolutional_layers = 3
n_dense_layers = 7
n_filt_pl = 10
n_filt_sz = 7

# Convergence
f_start_lr = 0.001
f_momentum = 0.9
f_decay_rate = 0.95
n_lr_gstep = 2000
n_lr_dstep = 1000

# Training
f_use_for_validation = 0.0125
f_use_for_inference = 0.1
f_precision_save_threshold = 0.85
f_reinitialization_threshold = 0.25
n_iterations_for_reinitialize = 1000000
n_max_iterations = 7000
n_iterations_for_stop = 5000
n_iterations_for_sleep = 2500

# Plotting
b_print_output = 1
n_plot_interval = 100

# Debugging
b_check_fitting = 0
n_check_fitting = 40


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
    n_classes = outputs_t.shape[1]
    sz_set = inputs_t.shape[0]

    # For debugging, check if the train error converges to 0% for a very small subset
    if b_check_fitting:
        sz_set = n_check_fitting
        global f_use_for_validation
        global f_use_for_inference
        global n_batches
        f_use_for_validation = 0.1
        f_use_for_inference = 0.1
        n_batches = 1

    sz_validate = int(sz_set * f_use_for_validation)
    sz_inference = int(sz_set * f_use_for_inference)
    sz_train = int(sz_set - sz_validate - sz_inference)
    sz_batch = int(sz_train / n_batches)
    sz_input = inputs_t.shape[1]

    # Trick tensorflow into knowing the size of _y tensor
    inputs = np.zeros([sz_set, sz_input], dtype=float, order='C')
    outputs = np.zeros([sz_set, n_classes], dtype=float, order='C')
    for i in range(sz_set):
        for j in range(n_classes):
            outputs[j] = outputs_t[j]
        for j in range(sz_input):
            inputs[i][j] = inputs_t[i][j]

    gc.collect()

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
    # Run Training
    ###########################################################################

    # Create Training Method
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    # Create Gradient Descent Optimizer
    f_learning_rate = tf.train.exponential_decay(f_start_lr, n_lr_gstep, n_lr_dstep, f_decay_rate, staircase=True)
    train_step = tf.train.MomentumOptimizer(f_learning_rate,
                                            f_momentum,
                                            use_locking=False,
                                            use_nesterov=True).minimize(cost_function)

    # Create Validation / Inference Method
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create Session
    sess = tf.InteractiveSession()

    # Create Variable Saver
    model_saver = tf.train.Saver()

    # Run Training
    tf.initialize_all_variables().run()

    t_start = time.time()

    # Used for automated training
    n_iteration_current = 0
    n_iterations_since_max_update = 0
    n_iterations_since_sleep = 0
    f_max_precision = 0.0

    # Used for real time plotting
    x_iteration_count = list()
    y_train_error = list()
    y_validate_error = list()

    b_stop = 0
    while b_stop == 0:
        for i in range(0, n_batches):

            # Use a mechanism to let the CPU and GPU cool when doing long training
            n_iterations_since_sleep += 1
            if n_iterations_since_sleep == n_iterations_for_sleep:
                time.sleep(60)  # Sleep for a little to let CPU and GPU cool
                n_iterations_since_sleep = 0

            # Get a random batch
            n_batch = rng.randint(0, n_batches - 1)
            n_iteration_current += 1

            if n_iteration_current == n_max_iterations:
                b_stop = 1

            batch_xs = x_train[(n_batch * sz_batch):((n_batch + 1) * sz_batch)][:]
            batch_ys = y_train[(n_batch * sz_batch):((n_batch + 1) * sz_batch)][:]
            sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})

            # Validation
            f_acc = sess.run(accuracy, feed_dict={x_: x_validate, y_: y_validate})
            f_train_acc = sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys})

            x_iteration_count.append(n_iteration_current)
            y_train_error.append(1 - f_train_acc)
            y_validate_error.append(1 - f_acc)

            # Display output in console
            if b_print_output:
                print("Iter.: %d; Validation: %.8f Training: %.8f" % (n_iteration_current, f_acc, f_train_acc))

            f_precision = f_acc
            if b_check_fitting:
                f_precision = f_train_acc

            # Stop training when the error is not decreasing for a certain number of iterations.
            if f_precision > f_max_precision:
                n_iterations_since_max_update = 0
                f_max_precision = f_precision

                # Save the model
                s_path = s_model_save_dir + str(f_precision) + "_" + str(n_iteration_current) + ".ckpt"
                model_save_path = model_saver.save(sess=sess, save_path=s_path)
                print("Model saved in file: %s" % model_save_path)

                # TODO
                # Add per class statistics to aide debugging

            else:
                n_iterations_since_max_update += 1
                if n_iterations_since_max_update == n_iterations_for_stop:
                    b_stop = 1
                    break

    t_stop = time.time()
    print("Training time        : " + str(t_stop - t_start))

    ###########################################################################
    # Run Inference
    ###########################################################################

    f_model_accuracy = sess.run(accuracy, feed_dict={x_: x_inference, y_: y_inference})
    print("Inference accuracy   : " + str(f_model_accuracy))


if __name__ == '__main__':

    tf.app.run()
