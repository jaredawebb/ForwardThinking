import tensorflow as tf
import numpy as np

# ToDo:  Save output layer once third layer is trained

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

batch_size = 128
num_classes = 10
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_sacols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    rotation_range=7,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    zoom_range=.1)

#x_train = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
#x_test = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)

datagen.fit(x_train)
images = datagen.flow(x_train, y_train, batch_size=batch_size)

################ Train the first layer  ######################
weights = []
train_accuracies = []
forward_accuracies = []
epoch_iter = len(x_train) // batch_size
epoch_sequence = [1,1,1]

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

keep_prob1 = tf.placeholder(tf.float32, shape=[])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

keep_prob2 = tf.placeholder(tf.float32, shape=[])
y_conv_drop = tf.nn.dropout(y_conv, keep_prob2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

flag = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[0]):
        # batch = mnist.train.next_batch(50)
        batch = images.next()
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                      y_: batch[1],
                                                      keep_prob1: 1., 
                                                      keep_prob2: 1.})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            
            acc1 = accuracy.eval(feed_dict={x: x_test[:1000].reshape((1000, 784)), y_: y_test[:1000], keep_prob1:1., keep_prob2:1.})
            acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000].reshape((1000, 784)), y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1.})
            acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000].reshape((1000, 784)), y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1.})
            acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000].reshape((1000, 784)), y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1.})
            acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000].reshape((1000, 784)), y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1.})
            acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000].reshape((1000, 784)), y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1.})
            acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000].reshape((1000, 784)), y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1.})
            acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000].reshape((1000, 784)), y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1.})
            acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000].reshape((1000, 784)), y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1.})
            acc10 = accuracy.eval(feed_dict={x: x_test[9000:].reshape((1000, 784)), y_: y_test[9000:], keep_prob1:1., keep_prob2:1.})
            
            train_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
            if flag:
                forward_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
                
        train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1],
                                  keep_prob1:0.3, keep_prob2:0.5})

    #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,
    #                                                  y_: mnist.test.labels}))
    
        if i == epoch_iter-1:
            weights.append([W_conv1.eval(), b_conv1.eval()])
            flag = False
    np.save('accuracies_layer1_aug', train_accuracies)
    print(len(forward_accuracies)) 
################ Train the second layer  ######################

train_accuracies = []                                    
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = tf.constant(weights[0][0])#weight_variable([3, 3, 1, 256])
b_conv1 = tf.constant(weights[0][1])

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

keep_prob1 = tf.placeholder(tf.float32, shape=[])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

keep_prob2 = tf.placeholder(tf.float32, shape=[])
y_conv_drop = tf.nn.dropout(y_conv, keep_prob2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

flag = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[0]):
        # batch = mnist.train.next_batch(50)
        batch = images.next()
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                      y_: batch[1],
                                                      keep_prob1: 1., 
                                                      keep_prob2: 1.})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            
            acc1 = accuracy.eval(feed_dict={x: x_test[:1000].reshape((1000, 784)), y_: y_test[:1000], keep_prob1:1., keep_prob2:1.})
            acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000].reshape((1000, 784)), y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1.})
            acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000].reshape((1000, 784)), y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1.})
            acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000].reshape((1000, 784)), y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1.})
            acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000].reshape((1000, 784)), y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1.})
            acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000].reshape((1000, 784)), y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1.})
            acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000].reshape((1000, 784)), y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1.})
            acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000].reshape((1000, 784)), y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1.})
            acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000].reshape((1000, 784)), y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1.})
            acc10 = accuracy.eval(feed_dict={x: x_test[9000:].reshape((1000, 784)), y_: y_test[9000:], keep_prob1:1., keep_prob2:1.})
            
            train_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
            if flag:
                forward_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
                
        train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1],
                                  keep_prob1:0.3, keep_prob2:0.5})

        if i == epoch_iter-1:
            weights.append([W_conv2.eval(), b_conv2.eval()])
            flag = False
    np.save('accuracies_layer2_aug', train_accuracies)
    print(len(forward_accuracies))
################ Train the third layer  ######################

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

keep_prob1 = tf.placeholder(tf.float32, shape=[])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

W_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

keep_prob2 = tf.placeholder(tf.float32, shape=[])
y_conv_drop = tf.nn.dropout(y_conv, keep_prob2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

flag = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[0]):
        # batch = mnist.train.next_batch(50)
        batch = images.next()
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                      y_: batch[1],
                                                      keep_prob1: 1., 
                                                      keep_prob2: 1.})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            
            acc1 = accuracy.eval(feed_dict={x: x_test[:1000].reshape((1000, 784)), y_: y_test[:1000], keep_prob1:1., keep_prob2:1.})
            acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000].reshape((1000, 784)), y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1.})
            acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000].reshape((1000, 784)), y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1.})
            acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000].reshape((1000, 784)), y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1.})
            acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000].reshape((1000, 784)), y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1.})
            acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000].reshape((1000, 784)), y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1.})
            acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000].reshape((1000, 784)), y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1.})
            acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000].reshape((1000, 784)), y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1.})
            acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000].reshape((1000, 784)), y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1.})
            acc10 = accuracy.eval(feed_dict={x: x_test[9000:].reshape((1000, 784)), y_: y_test[9000:], keep_prob1:1., keep_prob2:1.})
            
            train_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
            if flag:
                forward_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))
                
        train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1],
                                  keep_prob1:0.3, keep_prob2:0.5})
        if i == epoch_iter-1:
            weights.append([W_conv3.eval(), b_conv3.eval()])
            weights.append([W_fc1.eval(), b_fc1.eval()])
            weights.append([W_fc2.eval(), b_fc2.eval()])
            flag = False
    np.save('accuracies_layer3_aug', train_accuracies)
    print(len(forward_accuracies))
    

################################ Now that we have trained 3 layers, let's retrain each layer one at a time.

gibbs_epochs=10
import gibbs_utils

for i in range(gibbs_epochs):
    print(i)
    if i % 3 == 0:
        gibbs_utils.layer_1(weights, images, forward_accuracies, epoch_iter, mnist, learning_rates=[0.005])
    elif i % 3 == 1:
        gibbs_utils.layer_2(weights, images, forward_accuracies, epoch_iter, mnist, learning_rates=[0.005])
    elif i % 3 == 2:
        gibbs_utils.layer_3(weights, images, forward_accuracies, epoch_iter, mnist, learning_rates=[0.005])
    #ielif i % 9 == 3:
    #    gibbs_utils.layer_4(weights, images, forward_accuracies, epoch_iter, mnist)       
    #elif i % 9 == 4:
    #    gibbs_utils.layer_5(weights, images, forward_accuracies, epoch_iter, mnist)
    #elif i % 9 == 5:
    #    gibbs_utils.layer_4(weights, images, forward_accuracies, epoch_iter, mnist)
    #elif i % 9 == 6:
    #    gibbs_utils.layer_3(weights, images, forward_accuracies, epoch_iter, mnist)
    #elif i % 9 == 7:
    #    gibbs_utils.layer_2(weights, images, forward_accuracies, epoch_iter, mnist)
    #elif i % 9 == 8:
    #    gibbs_utils.layer_1(weights, images, forward_accuracies, epoch_iter, mnist)       

gibbs_utils.layer_3(weights, images, forward_accuracies, epoch_iter, mnist, mult=87, learning_rates=[0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005])

print(forward_accuracies[-10:])
print(np.mean(forward_accuracies[-10:]))

np.save('accuracies_gibbs', forward_accuracies)
np.save('weights_gibbs', weights)
    
