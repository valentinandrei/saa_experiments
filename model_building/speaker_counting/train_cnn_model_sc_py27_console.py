import time
import numpy as np
import tensorflow as tf
import random as rng
import gc

# Inputs
x_filename = '/home/valentin/Working/phd_project/build_dataset/scripts/x_train_normalized.txt'
y_filename = '/home/valentin/Working/phd_project/build_dataset/scripts/y_train.txt'
s_model_save_dir = '/home/valentin/Working/saa_experiments/model_building/models/speaker_counting/'

# Optimal parameters
# 1: LR 0.01, BS 400, Decay 0.98 -> 100% on debug data and altered real data

# Dataset Splitting
n_batches = 200

# Architecture
n_convolutional_layers = 5
n_filt_pl = 50
n_filt_sz = 20

# Convergence
f_start_lr = 0.1
f_momentum = 0.1
f_decay_rate = 0.96
n_lr_gstep = 2000
n_lr_dstep = 1000

# Training
f_use_for_validation = 0.0125
f_use_for_inference = 0.1
f_precision_save_threshold = 0.85
f_reinitialization_threshold = 0.25
n_iterations_for_reinitialize = 1000000
n_iterations_for_stop = 1000000
n_iterations_for_sleep = 1000000

# Plotting
b_print_output = 1
n_plot_interval = 100

# Debugging
b_check_fitting = 0
n_check_fitting = 60


# Generates a set of test data to check the neural network architecture health
def gen_debug_data():

    t_start = time.time()

    n_train_samples = 20000
    n_input_sz = 200
    n_classes = 5

    n_total_samples = n_train_samples
    inputs = np.zeros([n_total_samples, n_input_sz], dtype=float, order='C')
    outputs = np.zeros([n_total_samples, n_classes], dtype=float, order='C')

    for i in range(n_total_samples):
        id_class = rng.randint(0, n_classes - 1)
        outputs[i][id_class] = 1.0

        for j in range(n_input_sz):
            inputs[i][j] = (0.18 * (id_class + 1) - 0.5) + (1 / rng.randint(10, 15))

    t_stop = time.time()
    print("Data Generating Time (seconds): ", str(t_stop - t_start))

    return [inputs, outputs]


def main(_):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    # [inputs_t, outputs_t] = gen_debug_data()

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
            outputs[i][j] = outputs_t[i][j]
        for j in range(sz_input):
            inputs[i][j] = inputs_t[i][j]

    gc.collect()

    # Debug Code - Alter Inputs
    # for i in range(sz_set):
    #    n_speakers = 1 + np.argmax(outputs[i][:])
    #    for j in range(sz_input):
    #        inputs[i][j] = (inputs[i][j] + n_speakers) / (n_classes + 1)

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
    # Debug NN Architecture
    ###########################################################################

    # w0 = tf.Variable(tf.random_normal([sz_input, 1], mean=0.0), dtype=tf.float32)
    # b0 = tf.Variable(tf.random_normal([n_classes], mean=0.0), dtype=tf.float32)
    # y = tf.sigmoid(tf.matmul(x_, w0) + b0)

    ###########################################################################
    # Targeted CNN Architecture
    ###########################################################################

    v_activations = []
    v_filters = []
    v_biases = []

    sz_input_decrease = n_filt_sz - 1

    w_preproc = tf.Variable(tf.random_normal([sz_input, sz_input], mean=0.0), dtype=tf.float32)
    b_preproc = tf.Variable(tf.random_normal([sz_input]), dtype=tf.float32)
    x_preproc = tf.sigmoid(tf.matmul(x_, w_preproc) + b_preproc)

    x_0 = tf.reshape(x_preproc, [-1, sz_input, 1])
    w_0 = tf.Variable(tf.random_normal([n_filt_sz, 1, n_filt_pl], mean=0.0), dtype=tf.float32)
    b_0 = tf.Variable(tf.random_normal([sz_input - sz_input_decrease, n_filt_pl]), dtype=tf.float32)
    x_10 = tf.nn.conv1d(x_0, w_0, stride=1, padding='VALID')
    x_1 = tf.sigmoid(x_10 + b_0)
    v_activations.append(x_1)

    for i in range(1, n_convolutional_layers):
        sz_input_new = sz_input - (i+1) * sz_input_decrease
        w_i = tf.Variable(tf.random_normal([n_filt_sz, n_filt_pl, n_filt_pl], mean=0.0), dtype=tf.float32)
        b_i = tf.Variable(tf.random_normal([sz_input_new, n_filt_pl]), dtype=tf.float32)
        x_i0 = tf.nn.conv1d(v_activations[i-1], w_i, stride=1, padding='VALID')
        x_i1 = tf.sigmoid(x_i0 + b_i)

        v_filters.append(w_i)
        v_biases.append(b_i)
        v_activations.append(x_i1)

    sz_input_new = sz_input - n_convolutional_layers * sz_input_decrease
    w_final = tf.Variable(tf.random_normal([sz_input_new * n_filt_pl, n_classes], mean=0.0), dtype=tf.float32)
    b_final = tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32)
    x_final_conv = tf.reshape(v_activations[n_convolutional_layers - 1], [-1, sz_input_new * n_filt_pl])
    y = tf.sigmoid(tf.matmul(x_final_conv, w_final) + b_final)

    ###########################################################################
    # Run Training
    ###########################################################################

    # Create Training Method
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    # Create Gradient Descent Optimizer
    f_learning_rate = tf.train.exponential_decay(f_start_lr, n_lr_gstep, n_lr_dstep, f_decay_rate, staircase=True)
    train_step = tf.train.MomentumOptimizer(f_learning_rate, f_momentum, use_locking=False,
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
    n_iterations_since_reinitialize = 0
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
                print("Iter.: %d; Validation: %.5f Training: %.5f" % (n_iteration_current, f_acc, f_train_acc))

            f_precision = f_acc
            if b_check_fitting:
                f_precision = f_train_acc

            # The training might get stuck in a local optimum. In this case we reset the weights.
            if n_iterations_since_reinitialize < n_iterations_for_reinitialize:
                n_iterations_since_reinitialize += 1
            else:
                if f_acc < f_reinitialization_threshold:
                    print("Reinitializing all variables due to poor weight initialization.")
                    tf.initialize_all_variables().run()
                    n_iterations_since_reinitialize = 0
                    n_iterations_since_max_update = 0
                else:
                    n_iterations_since_reinitialize = 0

            # Stop training when the error is not decreasing for a certain number of iterations.
            if f_precision > f_max_precision:
                n_iterations_since_max_update = 0
                f_max_precision = f_precision

                # Save the model
                s_path = s_model_save_dir + str(f_precision) + "_" + str(n_iteration_current) + ".ckpt"
                model_save_path = model_saver.save(sess=sess, save_path=s_path)
                print("Model saved in file: %s" % model_save_path)
            else:
                n_iterations_since_max_update += 1
                if n_iterations_since_max_update == n_iterations_for_stop:
                    if f_precision < f_precision_save_threshold:
                        print("Reinitializing all variables due to poor weight initialization.")
                        tf.initialize_all_variables().run()
                        n_iterations_since_max_update = 0
                        n_iterations_since_reinitialize = 0
                        break
                    else:
                        b_stop = 1
                        break

    t_stop = time.time()
    print("Training time        : " + str(t_stop - t_start))

    ###########################################################################
    # Run Inference
    ###########################################################################

    f_model_accuracy = sess.run(accuracy, feed_dict={x_: x_inference, y_: y_inference})
    print("Inference accuracy   : " + str(f_model_accuracy))

    # Save Model Parameters
    model_save_path = model_saver.save(sess=sess, save_path="/tmp/dn_classifier.ckpt")
    print("Model saved in file: %s" % model_save_path)


if __name__ == '__main__':

    tf.app.run()
