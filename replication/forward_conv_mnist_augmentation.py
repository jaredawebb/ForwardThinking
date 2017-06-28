import tensorflow as tf
import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape)#,
        #initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def full_relu(input, shape):
    weights = tf.get_variable("weights", shape)#,
                              #initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", [shape[1]],
                              initializer=tf.constant_initializer(0.0))
    return tf.nn.relu(tf.matmul(input, weights) + biases)

def const_relu(input, constant):
    weights = tf.get_variable("weights", constant[0].shape,
                              initializer=tf.constant_initializer(constant[0]))
    biases = tf.get_variable("biases", constant[1].shape,
                              initializer=tf.constant_initializer(constant[1]))
                             
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def max_pool_2x2(x):
    print('hey. Sup.')
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

# x_train = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
# x_test = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)

datagen.fit(x_train)
images = datagen.flow(x_train, y_train, batch_size=batch_size)

################ Train the first layer  ######################

weights = []
train_accuracies = []
forward_accuracies = []
epoch_iter = len(x_train) // batch_size
epoch_sequence = [1,1,98]

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

with tf.variable_scope("1layer1"):
    h_conv1 = conv_relu(x_image, [3, 3, 1, 256], [256])
    h_pool1 = max_pool_2x2(h_conv1)
    print(h_pool1)
    
    flat_dim = int(h_pool1.get_shape()[1]*h_pool1.get_shape()[2]*h_pool1.get_shape()[3])
    print(flat_dim)
    h_pool1_flat = tf.reshape(h_poo11, [-1, flat_dim])

with tf.variable_scope("1fullyconnected"):
    keep_prob1 = tf.placeholder(tf.float32, shape=[])
    h_poo11_drop = tf.nn.dropout(h_pool1_flat, keep_prob1)
    
    h_fc1 = full_relu(h_pool1_drop, [flat_dim, 150])

    keep_prob2 = tf.placeholder(tf.float32, shape=[])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)
    
with tf.variable_scope("1output"):
    y_conv = full_relu(h_fc1_drop, [150, 10])

learning_rate = tf.placeholder(tf.float32, shape=[])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '1layer1')

learning_rates = [0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005]

flag = True
lr = learning_rates[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[0]):
        batch = images.next()
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                      y_: batch[1],
                                                      learning_rate: lr,
                                                      keep_prob1: 1., 
                                                      keep_prob2: 1.})

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

            acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])
            
            train_accuracies.append(acc)
            if flag:
                forward_accuracies.append(np.mean(acc))
            
            print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, acc))
                
        train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1], learning_rate: lr,
                                  keep_prob1:0.3, keep_prob2:0.5})

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

        elif i == epoch_iter*40:
            lr = learning_rates[3]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[3]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())

        elif i == epoch_iter*60:
            lr = learning_rates[4]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[4]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())

        elif i == epoch_iter*80:
            lr = learning_rates[5]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[5]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())
 
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,
    #                                                  y_: mnist.test.labels}))
    
        if i == epoch_iter-1:
            weights.append((train_vars[0].eval(), train_vars[1].eval()))
            flag = False
    np.save('accuracies_layer1_aug', train_accuracies)
    print(len(forward_accuracies)) 
################ Train the second layer  ######################

train_accuracies = []                                    
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

                             
with tf.variable_scope("2layer1"):
    h_conv1 = const_relu(x_image, weights[0])
    h_pool1 = max_pool_2x2(h_conv1)

with tf.variable_scope("2layer2"):
    h_conv2 = conv_relu(h_pool1, [3, 3, 256, 256], [256])
    h_pool2 = max_pool_2x2(h_conv2)

flat_dim = int(h_pool2.get_shape()[1]*h_pool2.get_shape()[2]*h_pool2.get_shape()[3])

with tf.variable_scope("2fullyconnected"):
    h_pool2_flat = tf.reshape(h_pool2_drop, [-1, flat_dim])
    
    keep_prob1 = tf.placeholder(tf.float32, shape=[])
    h_pool2_drop = tf.nn.dropout(h_pool2_flat, keep_prob1)
    
    h_fc1 = full_relu(h_pool2_drop, [flat_dim, 150])

    keep_prob2 = tf.placeholder(tf.float32, shape=[])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)
    
with tf.variable_scope("2output"):
    y_conv = full_relu(h_fc1_drop, [150, 10])

learning_rate = tf.placeholder(tf.float32, shape=[])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '2layer2')
                             
learning_rates = [0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005]

flag = True
lr = learning_rates[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[1]):
        batch = images.next()
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                      y_: batch[1],
                                                      learning_rate: lr,
                                                      keep_prob1: 1., 
                                                      keep_prob2: 1.})

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
            
            acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])
            
            train_accuracies.append(acc)
            if flag:
                forward_accuracies.append(np.mean(acc))
            
            print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, acc))
                
        train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1], learning_rate: lr,
                                  keep_prob1:0.3, keep_prob2:0.5})

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

        elif i == epoch_iter*40:
            lr = learning_rates[3]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[3]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())

        elif i == epoch_iter*60:
            lr = learning_rates[4]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[4]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())

        elif i == epoch_iter*80:
            lr = learning_rates[5]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[5]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())
        
        if i == (epoch_iter - 1):
            weights.append((train_vars[0].eval(), train_vars[1].eval()))
            flag = False
    np.save('accuracies_layer2_aug', train_accuracies)
    print(len(forward_accuracies))
################ Train the third layer  ######################

train_accuracies = []
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

with tf.variable_scope("3layer1"):
    h_conv1 = const_relu(x_image, weights[0])
    h_pool1 = max_pool_2x2(h_conv1)

with tf.variable_scope("3layer2"):
    h_conv2 = const_relu(h_pool1, weights[1])
    h_pool2 = max_pool_2x2(h_conv2)

with tf.variable_scope("3layer3"):
    h_conv3 = conv_relu(h_pool2, [3, 3, 256, 128], [128])
    h_pool3 = max_pool_2x2(h_conv3)
    
flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

with tf.variable_scope("3fullyconnected"):
    h_pool3_flat = tf.reshape(h_pool3_drop, [-1, flat_dim])
    
    keep_prob1 = tf.placeholder(tf.float32, shape=[])
    h_pool3_drop = tf.nn.dropout(h_pool3_flat, keep_prob1)
    
    h_fc1 = full_relu(h_pool3_drop, [flat_dim, 150])

    keep_prob2 = tf.placeholder(tf.float32, shape=[])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)
    
with tf.variable_scope("3output"):
    y_conv = full_relu(h_fc1_drop, [150, 10])

print(y_conv_drop.get_shape())
    
learning_rate = tf.placeholder(tf.float32, shape=[])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv_drop, labels=y_))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learning_rates = [0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005]

flag = True
lr = learning_rates[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch_iter*epoch_sequence[2]):
        batch = images.next()
        if i%100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                      y_: batch[1],
                                                      learning_rate: lr,
                                                      keep_prob1: 1., 
                                                      keep_prob2: 1.})

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
            
            acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])
            
            train_accuracies.append(acc)
            if flag:
                forward_accuracies.append(np.mean(acc))
            
            print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, acc))
                
        train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)), y_: batch[1], learning_rate: lr,
                                  keep_prob1:0.3, keep_prob2:0.5})

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

        elif i == epoch_iter*40:
            lr = learning_rates[3]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[3]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())

        elif i == epoch_iter*60:
            lr = learning_rates[4]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[4]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())

        elif i == epoch_iter*80:
            lr = learning_rates[5]
            print("Learning Rate Updated to: " + str(lr))
            #train_step = tf.train.AdamOptimizer(learning_rates[5]).minimize(cross_entropy)
            #sess.run(tf.global_variables_initializer())

'''
################ Train the output layer  ######################

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

W_conv2 = tf.constant(weights[2][0])
b_conv2 = tf.constant(weights[2][1])

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
            forward_accuracies.append(np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]))

        
        train_step.run(feed_dict={x: batch[0].reshape((50,784)), y_: batch[1]})

        #if i % 1100 == 0:
        #    weights.append((W_fc1.eval(), b_fc1.eval()))
    
    weights.append((W_fc1.eval(), b_fc1.eval()))
    np.save('accuracies_layer4_aug', train_accuracies)
    print(len(forward_accuracies))
'''    
np.save('accuracies_aug', forward_accuracies)
np.save('weights_aug', weights)
