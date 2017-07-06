import tensorflow as tf
import numpy as np
import sys
import time

import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

def decay_steps(base, total_iter, start_rate, final_rate):
    return int(total_iter*np.log10(base)/(np.log10(final_rate) - np.log10(start_rate)))

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape)

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
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

batch_size = 32
num_classes = 10
img_rows, img_cols = 32, 32
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

'''
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_sacols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)
'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
images = datagen.flow(x_train, y_train, batch_size=batch_size)

epochs = 200
epoch_iter = len(x_train) // batch_size
total_iter = epoch_iter*epochs
starter_learning_rate = 0.0005
final_learning_rate = starter_learning_rate/100
technique = 1

#architectures = [[64, 64], [128, 64], [128, 128], [256, 128], [256, 256], [512, 256], [512, 512],
#                 [64, 64, 64, 64], [128, 128, 64, 64], [128, 128, 128, 128], [256, 256, 128, 128], [256, 256, 256, 256],
#                 [512, 512, 256, 256], [512, 512, 512, 512],
#                 [64, 64, 64, 64, 64, 64], [128, 128, 128, 128, 128, 128], [256, 256, 256, 256, 256, 256],
#                 [512, 512, 512, 512, 512, 512]]

architectures = [[64, 64, 64, 64, 64, 64], [128, 128, 128, 128, 128, 128], [256, 256, 256, 256, 256, 256],
                 [512, 512, 512, 512, 512, 512]]

for arch in architectures:

    tf.reset_default_graph()
    t1 = time.time()
    
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    #######Layer 1
    with tf.variable_scope("layer1"):
        h_conv1 = conv_relu(x, [3, 3, 3, 128], [128])

    #######Layer 2
    with tf.variable_scope("layer2"):
        h_conv2 = conv_relu(h_conv1, [3, 3, 128, 128], [128])
        h_pool2 = max_pool_2x2(h_conv2)

    if len(arch) == 2:
        flat_dim = int(h_pool2.get_shape()[1]*h_pool2.get_shape()[2]*h_pool2.get_shape()[3])

        #######Fully Connected Layer
        with tf.variable_scope("fullyconnected"):
            
            h_pool2_flat = tf.reshape(h_pool2, [-1, flat_dim])

            keep_prob1 = tf.placeholder(tf.float32, shape=[])
            h_drop2 = tf.nn.dropout(h_pool2, keep_prob1)

            h_fc1 = full_relu(h_drop2, [flat_dim, 512])

            keep_prob2 = tf.placeholder(tf.float32, shape=[])
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)

        #######Output Layer
        with tf.variable_scope("output"):
            y_conv = full_relu(h_fc1_drop, [512, 10])
            
        layers = ['layer1', 'layer2', 'fullyconnected', 'output']

    if len(arch) == 4:
        #######Layer 3
        with tf.variable_scope("layer3"):
            keep_prob1 = tf.placeholder(tf.float32, shape=[])
            h_drop2 = tf.nn.dropout(h_pool2, keep_prob1)
            
            h_conv3 = conv_relu(h_drop2, [3, 3, 128, 128], [128])

        #######Layer 4
        with tf.variable_scope("layer4"):
            h_conv4 = conv_relu(h_conv3, [3, 3, 128, 128], [128])
            h_pool4 = max_pool_2x2(h_conv4)

        flat_dim = int(h_pool4.get_shape()[1]*h_pool4.get_shape()[2]*h_pool4.get_shape()[3])

        #######Fully Connected Layer
        with tf.variable_scope("fullyconnected"):
            
            h_pool4_flat = tf.reshape(h_pool4, [-1, flat_dim])

            keep_prob2 = tf.placeholder(tf.float32, shape=[])
            h_drop4 = tf.nn.dropout(h_pool4, keep_prob2)

            h_fc1 = full_relu(h_drop4, [flat_dim, 512])

            keep_prob3 = tf.placeholder(tf.float32, shape=[])
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob3)

        #######Output Layer
        with tf.variable_scope("output"):
            y_conv = full_relu(h_fc1_drop, [512, 10])
            
        layers = ['layer1', 'layer2', 'layer3', 'layer4', 'fullyconnected', 'output']
    
    if len(arch) == 6:

        #######Layer 3
        with tf.variable_scope("layer3"):
            keep_prob1 = tf.placeholder(tf.float32, shape=[])
            h_drop2 = tf.nn.dropout(h_pool2, keep_prob1)
            
            h_conv3 = conv_relu(h_drop2, [3, 3, 128, 128], [128])

        #######Layer 4
        with tf.variable_scope("layer4"):
            h_conv4 = conv_relu(h_conv3, [3, 3, 128, 128], [128])
            h_pool4 = max_pool_2x2(h_conv4)
            keep_prob2 = tf.placeholder(tf.float32, shape=[])
            h_drop4 = tf.nn.dropout(h_pool4, keep_prob2)


        #######Layer 5
        with tf.variable_scope("layer5"):
            h_conv5 = conv_relu(h_drop4, [3, 3, 128, 128], [128])

        #######Layer 6
        with tf.variable_scope("layer6"):
            h_conv6 = conv_relu(h_conv5, [3, 3, 128, 128], [128])

        flat_dim = int(h_conv6.get_shape()[1]*h_conv6.get_shape()[2]*h_conv6.get_shape()[3])

        #######Fully Connected Layer
        with tf.variable_scope("fullyconnected"):
            h_conv6_flat = tf.reshape(h_conv6, [-1, flat_dim])

            keep_prob3 = tf.placeholder(tf.float32, shape=[])
            h_drop6 = tf.nn.dropout(h_conv6_flat, keep_prob3)

            h_fc1 = full_relu(h_conv6_flat, [flat_dim, 512])

            keep_prob4 = tf.placeholder(tf.float32, shape=[])
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob4)

        #######Output Layer
        with tf.variable_scope("output"):
            y_conv = full_relu(h_fc1_drop, [512, 10])

        layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'fullyconnected', 'output']
        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

    # Get trainable variables from each layer
    train_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer) for layer in layers]

    # Set up rate decay
    global_step = tf.Variable(0, trainable=False)

    base = 0.98
    decay_step = decay_steps(base, total_iter, starter_learning_rate, final_learning_rate)
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
                print(str(arch) + " Starting Epoch %d of %d" % (i // epoch_iter, epochs))
                #print("Starting Epoch %d of %d, Training Layer %d" % (i // epoch_iter, epochs, epoch_number & len(train_steps))

            if i%100 == 0:
                
                if len(arch) == 2:
                    feed_dict = {x:batch[0], y_: batch[1], keep_prob1: 1., keep_prob2: 1.}
                elif len(arch) == 4:
                    feed_dict = {x:batch[0], y_: batch[1], keep_prob1: 1., keep_prob2: 1., keep_prob3: 1.}
                elif len(arch) == 6:
                    feed_dict = {x:batch[0], y_: batch[1], keep_prob1: 1., keep_prob2: 1., keep_prob3: 1., keep_prob4: 1.}
                
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                                
                test_acc = []
                chunk_size=100
                for j in range(0, len(x_test), chunk_size):
                    
                    if len(arch) == 2:
                        feed_dict = {x:x_test[j:j+chunk_size], y_: y_test[j:j+chunk_size], keep_prob1: 1., keep_prob2: 1.}
                    elif len(arch) == 4:
                        feed_dict = {x:x_test[j:j+chunk_size], y_: y_test[j:j+chunk_size],
                                     keep_prob1: 1., keep_prob2: 1., keep_prob3: 1.}
                    elif len(arch) == 6:
                        feed_dict = {x:x_test[j:j+chunk_size], y_: y_test[j:j+chunk_size],
                                     keep_prob1: 1., keep_prob2: 1., keep_prob3: 1., keep_prob4: 1.}
                    
                    test_acc.append(accuracy.eval(feed_dict=feed_dict))

                acc = np.mean(test_acc)
                
                curr_time = time.time()
                accuracies.append((acc, curr_time - t1))
                
                print("step %d, training accuracy %g, testing accuracy %g, learning rate %g, time %g" %(i, train_accuracy, acc, learning_rate.eval(), curr_time - t1))
                
                sys.stdout.flush()
                
            
            if len(arch) == 2:
                feed_dict = {x:batch[0], y_: batch[1], keep_prob1: 0.3, keep_prob2: 0.5}
            elif len(arch) == 4:
                feed_dict = {x:batch[0], y_: batch[1], keep_prob1: 0.3, keep_prob2: 0.3, keep_prob3: 0.5}
            elif len(arch) == 6:
                feed_dict = {x:batch[0], y_: batch[1], keep_prob1: 0.3, keep_prob2: 0.3, keep_prob3: 0.3, keep_prob4: 0.5}
            
            if technique == 0:
                train_steps[i % len(train_steps)].run(feed_dict=feed_dict)
            elif technique == 1:
                global_train_step.run(feed_dict=feed_dict)
                
            elif technique == 2:
                if epoch_number == 0:
                    train_steps[0].run(feed_dict=feed_dict)
                elif epoch_number == 1:
                    train_steps[1].run(feed_dict=feed_dict)
                else:
                    train_steps[-1].run(feed_dict=feed_dict)
            elif technique == 3:
                train_steps[epoch_number % len(train_steps)].run(feed_dict=feed_dict)
                
        title_string = './cifar_exp_results/backprop_accuracies'
        for size in arch:
            title_string += '_' + str(size)
        np.save(title_string, accuracies)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    