import tensorflow as tf
import numpy as np
import sys
import time

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

def decay_steps(base, total_iter, start_rate, final_rate):
    return int(total_iter*np.log10(base)/(np.log10(final_rate) - np.log10(start_rate)))

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

technique = 1

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

epochs = 100
epoch_iter = len(x_train) // batch_size
total_iter = epoch_iter*epochs

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

architectures = [[64, 64], [128, 64], [128, 128], [256, 128], [256, 256], [512, 256], [512, 512],
                 [64, 64, 32], [128, 64, 64], [128, 128, 64], [256, 128, 128], [256, 256, 128],
                 [512, 256, 256], [512, 512, 256]]


for arch in architectures:

    # Get start time
    t1 = time.time()
    
    # Set up architecture
    with tf.variable_scope("layer1"):
        h_conv1 = conv_relu(x_image, [3, 3, 1, arch[0]], [arch[0]])
        h_pool1 = max_pool_2x2(h_conv1)


    with tf.variable_scope("layer2"):
        h_conv2 = conv_relu(h_pool1, [3, 3, arch[0], arch[1]], arch[1])
        h_pool2 = max_pool_2x2(h_conv2)
        
    if len(arch) == 2:
        layers = ['layer1', 'layer2', 'fullyconnected', 'output']
        flat_dim = int(h_pool2.get_shape()[1]*h_pool2.get_shape()[2]*h_pool2.get_shape()[3])

        with tf.variable_scope("fullyconnected"):
            h_pool2_flat = tf.reshape(h_pool2, [-1, flat_dim])

            keep_prob1 = tf.placeholder(tf.float32, shape=[])
            h_pool2_drop = tf.nn.dropout(h_pool2_flat, keep_prob1)

            h_fc1 = full_relu(h_pool2_drop, [flat_dim, 150])

            keep_prob2 = tf.placeholder(tf.float32, shape=[])
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)
    else:
        layers = ['layer1', 'layer2', 'layer3', 'fullyconnected', 'output']
        with tf.variable_scope("layer3"):
            h_conv3 = conv_relu(h_pool2, [3, 3, arch[1], arch[2]], arch[2])
            h_pool3 = max_pool_2x2(h_conv3)
        
        flat_dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])

        with tf.variable_scope("fullyconnected"):
            h_pool3_flat = tf.reshape(h_pool3, [-1, flat_dim])

            keep_prob1 = tf.placeholder(tf.float32, shape=[])
            h_pool3_drop = tf.nn.dropout(h_pool3_flat, keep_prob1)

            h_fc1 = full_relu(h_pool3_drop, [flat_dim, 150])

            keep_prob2 = tf.placeholder(tf.float32, shape=[])
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)

    with tf.variable_scope("output"):
        y_conv = full_relu(h_fc1_drop, [150, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

    # Get trainable variables from each layer
    train_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer) for layer in layers]

    # Set up rate decay
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.005
    base = 0.98
    decay_step = decay_steps(base, total_iter, starter_learning_rate, starter_learning_rate/100)
    print("Decay step: " + str(decay_step))
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               decay_step, base, staircase=False)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_steps = [optimizer.minimize(cross_entropy,
                                      var_list=train_vars[i] + train_vars[-2] + train_vars[-1],
                                      global_step=global_step) for i in range(len(layers)-2)]
    
    global_train_step = optimizer.minimize(cross_entropy, global_step=global_step)
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)
        accuracies = []    
        for i in range(epoch_iter*epochs):
            
            epoch_number = i // epoch_iter
            
            batch = images.next()

            if i % epoch_iter == 0:
                print("Starting Epoch %d of %d" % (i // epoch_iter, epochs))
                #print("Starting Epoch %d of %d, Training Layer %d" % (i // epoch_iter, epochs, epoch_number & len(train_steps))

            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0].reshape((len(batch[0]), 784)), 
                                                          y_: batch[1],
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
                
                curr_time = time.time()
                accuracies.append((acc, curr_time - t1))
                
                print("step %d, training accuracy %g, testing accuracy %g, learning rate %g, time %g" %(i, train_accuracy, acc, learning_rate.eval(), curr_time - t1))
                
                sys.stdout.flush()
                
            if technique == 0:
                train_steps[i % len(train_steps)].run(feed_dict={x: batch[0].reshape((len(batch[0]),784)),
                                                                                y_: batch[1],
                                                                                keep_prob1:0.3,
                                                                                keep_prob2:0.5})
            elif technique == 1:
                global_train_step.run(feed_dict={x: batch[0].reshape((len(batch[0]),784)),
                                                                            y_: batch[1],
                                                                            keep_prob1:0.3,
                                                                            keep_prob2:0.5})
                
            elif technique == 2:
                if epoch_number == 0:
                    train_steps[0].run(feed_dict={x: batch[0].reshape((len(batch[0]),784)),
                                                                                y_: batch[1],
                                                                                keep_prob1:0.3,
                                                                                keep_prob2:0.5})
                elif epoch_number == 1:
                    train_steps[1].run(feed_dict={x: batch[0].reshape((len(batch[0]),784)),
                                                  y_: batch[1],
                                                  keep_prob1:0.3,
                                                  keep_prob2:0.5})
                else:
                    train_steps[-1].run(feed_dict={x: batch[0].reshape((len(batch[0]),784)),
                                                  y_: batch[1],
                                                  keep_prob1:0.3,
                                                  keep_prob2:0.5})
    
                
        title_string = 'accuracies'
        for size in arch:
            title_string += '_' + str(size)
        np.save('./mnist_exp_results/accuracies_'+str(cutoff), accuracies)




















