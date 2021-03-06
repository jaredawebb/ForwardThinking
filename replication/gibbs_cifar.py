import tensorflow as tf  # Tanner best: 256, 256, 5,5 mid
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

epoch_iter = len(x_train) // batch_size

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#backprop: 128 (3,3) -> 128 (3,3) -> dropout(.3) -> maxpool(size=(2,2)) -> 128 (3,3) -> 128 (3,3) -> dropout(.3) -> maxpool(size=(2,2)) -> 128 (3,3) -> 128 (3,3) -> FC 512 -> dropout(.5) -> Softmax 10

#x_image = tf.reshape(x, [-1,32,32,3])
#######Layer 1
with tf.variable_scope("layer1"):
    h_conv1 = conv_relu(x, [3, 3, 3, 128], [128])

#######Layer 2
with tf.variable_scope("layer2"):
    h_conv2 = conv_relu(h_conv1, [3, 3, 128, 128], [128])
    h_pool2 = max_pool_2x2(h_conv2)
    keep_prob1 = tf.placeholder(tf.float32, shape=[])
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob1)
    

#######Layer 3
with tf.variable_scope("layer3"):
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
    
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'fullyconnected', 'output']
train_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer) for layer in layers]

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           12000, 0.98, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate)

train_steps = [optimizer.minimize(cross_entropy,
                                  var_list=train_vars[i] + train_vars[-2] + train_vars[-1],
                                  global_step=global_step) for i in range(len(layers)-2)]

train_step = optimizer.minimize(cross_entropy, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.global_variables_initializer()

epochs = 200
cutoffs = [100, 150, 200]
choice = 1
for cutoff in cutoffs:
    with tf.Session() as sess:
        sess.run(init_op)
        accuracies = []
        #logs_path = '~/Documents/ForwardThinking/replication/logs/'
        #writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
        for i in range(epoch_iter*epochs):
            
            epoch_number = i // epoch_iter
            
            batch = images.next()

            if i % epoch_iter == 0:
                print("Starting Epoch %d of %d, cutoff %d" % (i // epoch_iter, epochs, cutoff))
                #print("Starting Epoch %d of %d, Training Layer %d" % (i // epoch_iter, epochs, epoch_number // len(train_steps))

            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], 
                                                          y_: batch[1],
                                                          keep_prob1: 1., 
                                                          keep_prob2: 1.,
                                                          keep_prob3: 1.,
                                                          keep_prob4: 1.})

                acc1 = accuracy.eval(feed_dict={x: x_test[:1000], y_: y_test[:1000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000], y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000], y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000], y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000], y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000], y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000], y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000], y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000], y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})
                acc10 = accuracy.eval(feed_dict={x: x_test[9000:], y_: y_test[9000:], keep_prob1:1., keep_prob2:1., keep_prob3:1., keep_prob4:1.})

                acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])

                accuracies.append(acc)

                print("step %d, training accuracy %g, testing accuracy %g, learning rate %g"%(i, train_accuracy, acc, learning_rate.eval()))
            #print([np.max(weight[0].eval()) for weight in train_vars])


            
            if choice == 0:
                if i < cutoff*epoch_iter:
                    train_step.run(feed_dict={x: batch[0], y_: batch[1],
                                          keep_prob1:0.3, keep_prob2:0.3, keep_prob3:0.3, keep_prob4:.5})

            #elif choice == 1:
            else:
                if i < cutoff*epoch_iter:
                    train_steps[i % len(train_steps)].run(feed_dict={x: batch[0],
                                                                               y_: batch[1],
                                                                                keep_prob1:0.3,
                                                                                keep_prob2:0.3,
                                                                                keep_prob3:0.3,
                                                                                keep_prob4:0.5})

                    #train_steps[epoch_number % len(train_steps)].run(feed_dict={x: batch[0],
                    #                                                            y_: batch[1],
                    #                                                            keep_prob1:0.3,
                    #                                                            keep_prob2:0.3,
                    #                                                            keep_prob3:0.5})

                else:
                    if epoch_iter*cutoff == i:
                        print("Switching to output layer only.")
                    train_steps[-1].run(feed_dict={x: batch[0], y_: batch[1],
                                          keep_prob1:0.3, keep_prob2:0.3, keep_prob3:0.3, keep_prob4:0.5})
            sys.stdout.flush()
            
            '''
            if (len(accuracies) > 80):
                if (np.mean(accuracies[-80:]) - accuracies[-1]) < 0.0000001:
                    print(np.mean(accuracies
                    print("HEY!  STOP!")
                    break
            '''
        np.save('./results/decay_cifar_'+str(cutoff), accuracies)
