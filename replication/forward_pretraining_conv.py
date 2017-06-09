import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

weights = np.load('weights.npy', encoding='latin1')

################ Train the first layer  ######################
new_weights = []
train_accuracies = []
forward_accuracies = []
epoch_iter = 1100
epoch_sequence = [10,9,8,7]

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([3, 3, 1, 256])
b_conv1 = bias_variable([256])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

flat_dim = int(h_pool1.get_shape()[1]*h_pool1.get_shape()[2]*h_pool1.get_shape()[3])

W_fc1 = weight_variable([flat_dim, 150])
b_fc1 = bias_variable([150])

h_pool1_flat = tf.reshape(h_pool1, [-1, flat_dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, 0.3)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_conv_drop = tf.nn.dropout(y_conv, 0.5)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

flag = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[0]):
        batch = mnist.train.next_batch(50)
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], 
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
                
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,
    #                                                  y_: mnist.test.labels}))
    
        if i == 1100:
            new_weights.append((W_conv1.eval(), b_conv1.eval()))
            flag = False
    np.save('p_accuracies_layer1', train_accuracies)
    
################ Train the second layer  ######################

train_accuracies = []                                    
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = tf.Variable(weights[0][0])#weight_variable([3, 3, 1, 256])
b_conv1 = tf.Variable(weights[0][1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#######

W_conv2 = weight_variable([3, 3, 256, 256])
b_conv2 = bias_variable([256])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#######

flat_dim = int(h_pool2.get_shape()[1]*h_pool2.get_shape()[2]*h_pool2.get_shape()[3])

W_fc1 = weight_variable([flat_dim, 150])
b_fc1 = bias_variable([150])

h_pool2_flat = tf.reshape(h_pool2, [-1, flat_dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, 0.3)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_conv_drop = tf.nn.dropout(y_conv, 0.5)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

flag = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[1]):
        batch = mnist.train.next_batch(50)
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], 
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
                
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        if i == 1100:
            new_weights.append((W_conv2.eval(), b_conv2.eval()))
            flag = False
    np.save('p_accuracies_layer2', train_accuracies)
    
################ Train the third layer  ######################

train_accuracies = []
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = tf.Variable(weights[0][0])#weight_variable([3, 3, 1, 256])
b_conv1 = tf.Variable(weights[0][1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#######

W_conv2 = tf.Variable(weights[1][0])
b_conv2 = tf.Variable(weights[1][1])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#######

W_conv3 = weight_variable([3, 3, 256, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#######

flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

W_fc1 = weight_variable([flat_dim, 150])
b_fc1 = bias_variable([150])

h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, 0.3)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_conv_drop = tf.nn.dropout(y_conv, 0.5)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

flag = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[2]):
        batch = mnist.train.next_batch(50)
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], 
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
                
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        if i == 1100:
            new_weights.append((W_conv3.eval(), b_conv3.eval()))
            flag = False
    np.save('p_accuracies_layer3', train_accuracies)
    
################ Train the output layer  ######################

train_accuracies = []
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = tf.Variable(weights[0][0])#weight_variable([3, 3, 1, 256])
b_conv1 = tf.Variable(weights[0][1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#######

W_conv2 = tf.Variable(weights[1][0])
b_conv2 = tf.Variable(weights[1][1])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#######

W_conv2 = tf.Variable(weights[2][0])
b_conv2 = tf.Variable(weights[2][1])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#######

flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

W_fc1 = weight_variable([flat_dim, 150])
b_fc1 = bias_variable([150])

h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, 0.3)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_conv_drop = tf.nn.dropout(y_conv, 0.5)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

flag = True
with tf.Session() as sess:
    num_epochs=3
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[3]):
        batch = mnist.train.next_batch(50)
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], 
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
            forward_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))


        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        #if i % 1100 == 0:
        #    weights.append((W_fc1.eval(), b_fc1.eval()))
    
    new_weights.append((W_fc1.eval(), b_fc1.eval()))
    np.save('p_accuracies_layer4', train_accuracies)
    
    
np.save('p_accuracies', forward_accuracies)
np.save('new_weights', new_weights)
