import tensorflow as tf
import numpy as np

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

num_classes = 10
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_sacols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

y_test = keras.utils.to_categorical(y_test, num_classes)


def layer_1(weights, images, forward_accuracies, epoch_iter, mnist, learning_rates=[1e-4]):
    # Pass in the weights, freeze all but the first layer, and then update the weights
    
    train_accuracies = []
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = tf.Variable(weights[0][0])#weight_variable([3, 3, 1, 256])
    b_conv1 = tf.Variable(weights[0][1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #######

    W_conv2 = tf.constant(weights[1][0])
    b_conv2 = tf.constant(weights[1][1])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #######
    W_conv3 = tf.constant(weights[2][0])
    b_conv3 = tf.constant(weights[2][1])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #######

    flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

    W_fc1 = weight_variable([flat_dim, 150])
    b_fc1 = bias_variable([150])

    #W_fc1 = tf.constant(weights[3][0])#weight_variable([flat_dim, 150])
    #b_fc1 = tf.constant(weights[3][1])#bias_variable([150])

    h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    keep_prob1 = tf.placeholder(tf.float32, shape=[])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

    W_fc2 = weight_variable([150, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    keep_prob2 = tf.placeholder(tf.float32, shape=[])
    y_conv_drop = tf.nn.dropout(y_conv, keep_prob2)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = learning_rates[0]
    flag = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch_iter):
            # batch = mnist.train.next_batch(50)
            batch = images.next()
            if i%100 == 0 and i > 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                          y_: batch[1],
                                                          keep_prob1: 1., 
                                                          keep_prob2: 1.,
                                                          learning_rate: lr})

                acc1 = accuracy.eval(feed_dict={x: x_test[:1000].reshape((1000, 784)), y_: y_test[:1000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000].reshape((1000, 784)), y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000].reshape((1000, 784)), y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000].reshape((1000, 784)), y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000].reshape((1000, 784)), y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000].reshape((1000, 784)), y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000].reshape((1000, 784)), y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000].reshape((1000, 784)), y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000].reshape((1000, 784)), y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc10 = accuracy.eval(feed_dict={x: x_test[9000:].reshape((1000, 784)), y_: y_test[9000:], keep_prob1:1., keep_prob2:1., learning_rate:lr})

                acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])
                train_accuracies.append(acc)
                if flag:
                    forward_accuracies.append(acc)

                print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, acc))

            train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1],
                                      keep_prob1:0.3, keep_prob2:0.5, learning_rate:lr})

            if i == epoch_iter-1:
                weights[0][0] = W_conv1.eval()
                weights[0][1] = b_conv1.eval()
                flag = False
        
        
        
def layer_2(weights, images, forward_accuracies, epoch_iter, mnist, learning_rates=[1e-4]):
    # Pass in the weights, freeze all but the first layer, and then update the weights
    
    train_accuracies = []
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = tf.constant(weights[0][0])#weight_variable([3, 3, 1, 256])
    b_conv1 = tf.constant(weights[0][1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #######

    W_conv2 = tf.Variable(weights[1][0])
    b_conv2 = tf.Variable(weights[1][1])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #######
    W_conv3 = tf.constant(weights[2][0])
    b_conv3 = tf.constant(weights[2][1])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #######

    flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])
    
    W_fc1 = weight_variable([flat_dim, 150])
    b_fc1 = bias_variable([150])

    #W_fc1 = tf.constant(weights[3][0])#weight_variable([flat_dim, 150])
    #b_fc1 = tf.constant(weights[3][1])#bias_variable([150])

    h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    keep_prob1 = tf.placeholder(tf.float32, shape=[])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

    W_fc2 = weight_variable([150, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    keep_prob2 = tf.placeholder(tf.float32, shape=[])
    y_conv_drop = tf.nn.dropout(y_conv, keep_prob2)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = learning_rates[0]
    flag = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch_iter):
            # batch = mnist.train.next_batch(50)
            batch = images.next()
            if i%100 == 0 and i > 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                          y_: batch[1],
                                                          keep_prob1: 1., 
                                                          keep_prob2: 1.,
                                                          learning_rate: lr})

                acc1 = accuracy.eval(feed_dict={x: x_test[:1000].reshape((1000, 784)), y_: y_test[:1000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000].reshape((1000, 784)), y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000].reshape((1000, 784)), y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000].reshape((1000, 784)), y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000].reshape((1000, 784)), y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000].reshape((1000, 784)), y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000].reshape((1000, 784)), y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000].reshape((1000, 784)), y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000].reshape((1000, 784)), y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc10 = accuracy.eval(feed_dict={x: x_test[9000:].reshape((1000, 784)), y_: y_test[9000:], keep_prob1:1., keep_prob2:1., learning_rate:lr})

                acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])
                train_accuracies.append(acc)
                if flag:
                    forward_accuracies.append(acc)

                print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, acc))

            train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1],
                                      keep_prob1:0.3, keep_prob2:0.5, learning_rate:lr})

            if i == epoch_iter - 1:
                weights[1][0] = W_conv2.eval()
                weights[1][1] = b_conv2.eval()
                flag = False

def layer_3(weights, images, forward_accuracies, epoch_iter, mnist, mult=1, learning_rates=[1e-4]):
    # Pass in the weights, freeze all but the first layer, and then update the weights
    
    train_accuracies = []
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = tf.constant(weights[0][0])#weight_variable([3, 3, 1, 256])
    b_conv1 = tf.constant(weights[0][1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #######

    W_conv2 = tf.constant(weights[1][0])
    b_conv2 = tf.constant(weights[1][1])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #######
    W_conv3 = tf.Variable(weights[2][0])
    b_conv3 = tf.Variable(weights[2][1])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #######

    flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

    W_fc1 = weight_variable([flat_dim, 150])
    b_fc1 = bias_variable([150])

    #W_fc1 = tf.constant(weights[3][0])#weight_variable([flat_dim, 150])
    #b_fc1 = tf.constant(weights[3][1])#bias_variable([150])

    h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    keep_prob1 = tf.placeholder(tf.float32, shape=[])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

    W_fc2 = weight_variable([150, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    keep_prob2 = tf.placeholder(tf.float32, shape=[])
    y_conv_drop = tf.nn.dropout(y_conv, keep_prob2)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = learning_rates[0]
    flag = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch_iter*mult):
            # batch = mnist.train.next_batch(50)
            batch = images.next()
            if i%100 == 0 and i > 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                          y_: batch[1],
                                                          keep_prob1: 1., 
                                                          keep_prob2: 1.,
                                                          learning_rate: lr})


                acc1 = accuracy.eval(feed_dict={x: x_test[:1000].reshape((1000, 784)), y_: y_test[:1000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000].reshape((1000, 784)), y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000].reshape((1000, 784)), y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000].reshape((1000, 784)), y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000].reshape((1000, 784)), y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000].reshape((1000, 784)), y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000].reshape((1000, 784)), y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000].reshape((1000, 784)), y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000].reshape((1000, 784)), y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1., learning_rate:lr})
                acc10 = accuracy.eval(feed_dict={x: x_test[9000:].reshape((1000, 784)), y_: y_test[9000:], keep_prob1:1., keep_prob2:1., learning_rate:lr})

                acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])
                train_accuracies.append(acc)
                if flag:
                    forward_accuracies.append(acc)

                print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, acc))
            train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1],
                                      keep_prob1:0.3, keep_prob2:0.5, learning_rate:lr})
            if i == epoch_iter - 1 and mult == 1:
                weights[2][0] = W_conv3.eval()
                weights[2][1] = b_conv3.eval()
                flag = False
                
            if i == epoch_iter*2:
                lr = learning_rates[1]
                print("Learning Rate Updated to: " + str(lr))
                #train_step = tf.train.AdamOptimizer(learning_rates[1]).minimize(cross_entropy)
                #sess.run(tf.global_variables_initializer())

            elif i == epoch_iter*10:
                lr = learning_rates[2]
                print("Learning Rate Updated to: " + str(lr))
                #train_step = tf.train.AdamOptimizer(learning_rates[2]).minimize(cross_entropy)
                #sess.run(tf.global_variables_initializer())

            elif i == epoch_iter*30:
                lr = learning_rates[3]
                print("Learning Rate Updated to: " + str(lr))
                #train_step = tf.train.AdamOptimizer(learning_rates[3]).minimize(cross_entropy)
                #sess.run(tf.global_variables_initializer())

            elif i == epoch_iter*50:
                lr = learning_rates[4]
                print("Learning Rate Updated to: " + str(lr))
                #train_step = tf.train.AdamOptimizer(learning_rates[4]).minimize(cross_entropy)
                #sess.run(tf.global_variables_initializer())

            elif i == epoch_iter*70:
                lr = learning_rates[5]
                print("Learning Rate Updated to: " + str(lr))
                #train_step = tf.train.AdamOptimizer(learning_rates[5]).minimize(cross_entropy)
                #sess.run(tf.global_variables_initializer())

def layer_4(weights, images, forward_accuracies, epoch_iter, mnist):
    # Pass in the weights, freeze all but the first layer, and then update the weights
    
    train_accuracies = []
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = tf.constant(weights[0][0])#weight_variable([3, 3, 1, 256])
    b_conv1 = tf.constant(weights[0][1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #######

    W_conv2 = tf.constant(weights[1][0])
    b_conv2 = tf.constant(weights[1][1])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #######
    W_conv3 = tf.constant(weights[2][0])
    b_conv3 = tf.constant(weights[2][1])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #######

    flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

    W_fc1 = tf.Variable(weights[3][0])#weight_variable([flat_dim, 150])
    b_fc1 = tf.Variable(weights[3][1])#bias_variable([150])

    h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, 0.3)

    W_fc2 = tf.constant(weights[4][0])#weight_variable([150, 10])
    b_fc2 = tf.constant(weights[4][1])#bias_variables([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y_conv_drop = tf.nn.dropout(y_conv, 0.5)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    flag = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch_iter):
            batch = images.next()
            if i%100 == 0 and i > 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((50, 784)), 
                                                          y_: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                acc1 = accuracy.eval(feed_dict={x: mnist.test.images[:1000], y_: mnist.test.labels[:1000]})
                acc2 = accuracy.eval(feed_dict={x: mnist.test.images[1000:2000], y_: mnist.test.labels[1000:2000]})
                acc3 = accuracy.eval(feed_dict={x: mnist.test.images[2000:3000], y_: mnist.test.labels[2000:3000]})
                acc4 = accuracy.eval(feed_dict={x: mnist.test.images[3000:4000], y_: mnist.test.labels[3000:4000]})
                acc5 = accuracy.eval(feed_dict={x: mnist.test.images[4000:5000], y_: mnist.test.labels[4000:5000]})
                acc6 = accuracy.eval(feed_dict={x: mnist.test.images[5000:6000], y_: mnist.test.labels[5000:6000]})
                acc7 = accuracy.eval(feed_dict={x: mnist.test.images[6000:7000], y_: mnist.test.labels[6000:7000]})
                acc8 = accuracy.eval(feed_dict={x: mnist.test.images[7000:8000], y_: mnist.test.labels[7000:8000]})
                acc9 = accuracy.eval(feed_dict={x: mnist.test.images[8000:9000], y_: mnist.test.labels[8000:9000]})
                acc10 = accuracy.eval(feed_dict={x: mnist.test.images[9000:], y_: mnist.test.labels[9000:]})

                train_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
                if flag:
                    forward_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))

            train_step.run(feed_dict={x: batch[0].reshape((50,784)), y_: batch[1]})

            if i == 1099:
                weights[3][0] = W_fc1.eval()
                weights[3][1] = b_fc1.eval()
                flag = False

def layer_5(weights, images, forward_accuracies, epoch_iter, mnist):
    # Pass in the weights, freeze all but the first layer, and then update the weights
    
    train_accuracies = []
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = tf.constant(weights[0][0])#weight_variable([3, 3, 1, 256])
    b_conv1 = tf.constant(weights[0][1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #######

    W_conv2 = tf.constant(weights[1][0])
    b_conv2 = tf.constant(weights[1][1])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #######
    W_conv3 = tf.constant(weights[2][0])
    b_conv3 = tf.constant(weights[2][1])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #######

    flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

    W_fc1 = tf.constant(weights[3][0])#weight_variable([flat_dim, 150])
    b_fc1 = tf.constant(weights[3][1])#bias_variable([150])

    h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, 0.3)

    W_fc2 = tf.Variable(weights[4][0])#weight_variable([150, 10])
    b_fc2 = tf.Variable(weights[4][1])#bias_variables([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y_conv_drop = tf.nn.dropout(y_conv, 0.5)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    flag = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch_iter):
            batch = images.next()
            if i%100 == 0 and i > 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((50, 784)), 
                                                          y_: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                acc1 = accuracy.eval(feed_dict={x: mnist.test.images[:1000], y_: mnist.test.labels[:1000]})
                acc2 = accuracy.eval(feed_dict={x: mnist.test.images[1000:2000], y_: mnist.test.labels[1000:2000]})
                acc3 = accuracy.eval(feed_dict={x: mnist.test.images[2000:3000], y_: mnist.test.labels[2000:3000]})
                acc4 = accuracy.eval(feed_dict={x: mnist.test.images[3000:4000], y_: mnist.test.labels[3000:4000]})
                acc5 = accuracy.eval(feed_dict={x: mnist.test.images[4000:5000], y_: mnist.test.labels[4000:5000]})
                acc6 = accuracy.eval(feed_dict={x: mnist.test.images[5000:6000], y_: mnist.test.labels[5000:6000]})
                acc7 = accuracy.eval(feed_dict={x: mnist.test.images[6000:7000], y_: mnist.test.labels[6000:7000]})
                acc8 = accuracy.eval(feed_dict={x: mnist.test.images[7000:8000], y_: mnist.test.labels[7000:8000]})
                acc9 = accuracy.eval(feed_dict={x: mnist.test.images[8000:9000], y_: mnist.test.labels[8000:9000]})
                acc10 = accuracy.eval(feed_dict={x: mnist.test.images[9000:], y_: mnist.test.labels[9000:]})

                train_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
                if flag:
                    forward_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))

            train_step.run(feed_dict={x: batch[0].reshape((50,784)), y_: batch[1]})

            if i == 1099:
                weights[4][0] = W_fc2.eval()
                weights[4][1] = b_fc2.eval()
                flag = False
