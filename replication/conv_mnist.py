import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_accuracies = []

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
t
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

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([3, 3, 1, 256])
b_conv1 = bias_variable([256])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 256, 256])
b_conv2 = bias_variable([256])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 3, 256, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([4 * 4 * 128, 150])
b_fc1 = bias_variable([150])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob1 = tf.placeholder(tf.float32, shape=[])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

keep_prob2 = tf.placeholder(tf.float32, shape=[])
y_conv_drop = tf.nn.dropout(y_conv, keep_prob2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    num_epochs=5
    sess.run(tf.global_variables_initializer())
    
    for i in range(1100*num_epochs):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], 
                                                      y_: batch[1],
                                                      keep_prob1:1.,
                                                      keep_prob2:1.})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            train_accuracies.append(accuracy.eval(feed_dict={x: mnist.test.images,
                                                      y_: mnist.test.labels,
                                                            keep_prob1:1.,
                                                            keep_prob2:1.}))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob1:0.3, keep_prob2:0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,
                                                      y_: mnist.test.labels}))
   