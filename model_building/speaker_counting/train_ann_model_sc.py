import time
import numpy as np
import tensorflow as tf
import random as rng
import gc
import matplotlib.pyplot as plt
# import os

x_filename = '/home/valentin/Working/phd_project/build_dataset/scripts/x_train_normalized.txt'
y_filename = '/home/valentin/Working/phd_project/build_dataset/scripts/y_train.txt'
s_model_save_dir = '/home/valentin/Working/phd_project/build_model/'


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

    [inputs_t, outputs_t] = gen_debug_data()

    ###########################################################################
    # Load Train / Test Data
    ###########################################################################

    t_start = time.time()

    # Load Input Files
    # inputs_t = np.loadtxt(x_filename)
    # outputs_t = np.loadtxt(y_filename)

    # Experiment Parameters
    n_classes = outputs_t.shape[1]
    n_batches = 100
    sz_set = inputs_t.shape[0]

    sz_validate = int(sz_set * 0.05)
    sz_inference = int(sz_set * 0.1)
    sz_train = int(sz_set - sz_validate - sz_inference)
    sz_batch = int(sz_train / n_batches)
    sz_input = inputs_t.shape[1]

    # Trick python into knowing the size of _y tensor
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
    # Targeted NN Architecture
    ###########################################################################

    n_dense_layers = 20
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
    # Run Training
    ###########################################################################

    # Training parameters
    f_start_learning_rate = 0.1
    f_decay_rate = 0.98
    n_lr_global_step = 2000
    n_lr_decay_step = 1000

    f_precision_save_threshold = 0.4
    f_reinitialization_threshold = 0.25
    n_iterations_for_reinitialize = 500
    n_iterations_for_stop = 5000
    n_iterations_for_sleep = 100000

    # Create Training Method
    y_ = tf.placeholder(tf.float32, [None, n_classes])
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # cost_function = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # Create Gradient Descent Optimizer
    f_learning_rate = tf.train.exponential_decay(f_start_learning_rate, n_lr_global_step,
                                                 n_lr_decay_step, f_decay_rate, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(f_learning_rate).minimize(cost_function)

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
    n_plot_interval = 25
    x_iteration_count = list()
    y_train_error = list()
    y_validate_error = list()
    plt.xlabel("Iteration Count", fontsize=14)
    plt.ylabel("Training and Validation Error", fontsize=14)
    plt.ion()
    plt.ylim((0.0, 1.0))
    plt.grid()

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

            # Update plotting
            if n_iteration_current % n_plot_interval == 0:
                x_iteration_count.append(n_iteration_current)
                y_train_error.append(1 - f_train_acc)
                y_validate_error.append(1 - f_acc)
                plt.plot(x_iteration_count, y_train_error, color='red')
                plt.plot(x_iteration_count, y_validate_error, color='blue')
                plt.pause(0.05)

            print("Iter.: %d; validate accuracy: %.3f" % (n_iteration_current, f_acc))
            print("Iter.: %d; training accuracy: %.3f" % (n_iteration_current, f_train_acc))
            print("Iterations since max accuracy: %d" % n_iterations_since_max_update)
            print("Max accuracy: %.3f" % f_max_precision)
            print("**************************")
            print("")

            f_precision = f_acc

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
