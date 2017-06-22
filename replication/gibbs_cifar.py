import tensorflow as tf
import numpy as np

import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

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


batch_size = 128
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

#x_image = tf.reshape(x, [-1,32,32,3])
#######Layer 1
with tf.variable_scope("layer1"):
    W_conv1 = weight_variable([3, 3, 3, 32])#weight_variable([3, 3, 1, 256])
    b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

#######Layer 2
with tf.variable_scope("layer2"):
    W_conv2 = weight_variable([3, 3, 32, 32])
    b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
keep_prob1 = tf.placeholder(tf.float32, shape=[])
h_drop2 = tf.nn.dropout(h_pool2, keep_prob1)

#######Layer 3
with tf.variable_scope("layer3"):
    W_conv3 = weight_variable([3, 3, 32, 64])
    b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3) + b_conv3)

#######Layer 4
with tf.variable_scope("layer4"):
    W_conv4 = weight_variable([3, 3, 64, 64])
    b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)
keep_prob2 = tf.placeholder(tf.float32, shape=[])
h_drop4 = tf.nn.dropout(h_pool4, keep_prob2)

#######Layer 5
with tf.variable_scope("layer5"):
    W_conv5 = weight_variable([3, 3, 64, 128])
    b_conv5 = bias_variable([128])

h_conv5 = tf.nn.relu(conv2d(h_drop4, W_conv5) + b_conv5)

#######Layer 6
with tf.variable_scope("layer6"):
    W_conv6 = weight_variable([3, 3, 128, 128])
    b_conv6 = bias_variable([128])

h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

flat_dim = int(h_conv6.get_shape()[1]*h_conv6.get_shape()[2]*h_conv6.get_shape()[3])

with tf.variable_scope("fullyconnected"):
    W_fc1 = weight_variable([flat_dim, 256])
    b_fc1 = bias_variable([256])

h_conv6_flat = tf.reshape(h_conv6, [-1, flat_dim])
h_fc1 = tf.nn.relu(tf.matmul(h_conv6_flat, W_fc1) + b_fc1)

keep_prob3 = tf.placeholder(tf.float32, shape=[])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob3)

with tf.variable_scope("output"):
    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])
    
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'fullyconnected', 'output']
train_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer) for layer in layers]
optimizer = tf.train.AdamOptimizer(1e-4)

train_steps = [optimizer.minimize(cross_entropy,
                                  var_list=train_vars[i] + train_vars[-1]) for i in range(len(layers)-1)]
#train_step = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.global_variables_initializer()

epochs = 300
cutoffs = [16, 32, 64, 128, 256, 300]
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
                print("Starting Epoch %d of %d" % (i // epoch_iter, epochs))
                #print("Starting Epoch %d of %d, Training Layer %d" % (i // epoch_iter, epochs, epoch_number // len(train_steps))

            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], 
                                                          y_: batch[1],
                                                          keep_prob1: 1., 
                                                          keep_prob2: 1.,
                                                          keep_prob3: 1.})

                acc1 = accuracy.eval(feed_dict={x: x_test[:1000], y_: y_test[:1000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc2 = accuracy.eval(feed_dict={x: x_test[1000:2000], y_: y_test[1000:2000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc3 = accuracy.eval(feed_dict={x: x_test[2000:3000], y_: y_test[2000:3000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc4 = accuracy.eval(feed_dict={x: x_test[3000:4000], y_: y_test[3000:4000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc5 = accuracy.eval(feed_dict={x: x_test[4000:5000], y_: y_test[4000:5000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc6 = accuracy.eval(feed_dict={x: x_test[5000:6000], y_: y_test[5000:6000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc7 = accuracy.eval(feed_dict={x: x_test[6000:7000], y_: y_test[6000:7000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc8 = accuracy.eval(feed_dict={x: x_test[7000:8000], y_: y_test[7000:8000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc9 = accuracy.eval(feed_dict={x: x_test[8000:9000], y_: y_test[8000:9000], keep_prob1:1., keep_prob2:1., keep_prob3:1.})
                acc10 = accuracy.eval(feed_dict={x: x_test[9000:], y_: y_test[9000:], keep_prob1:1., keep_prob2:1., keep_prob3:1.})

                acc = np.mean([acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10])

                accuracies.append(acc)

                print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, acc))
            #print(np.max(W_conv1.eval()))

            #train_step.run(feed_dict={x: batch[0], y_: batch[1],
            #                          keep_prob1:0.5, keep_prob2:0.5, keep_prob3:0.5})
            if i < cutoff*epoch_iter:
                train_steps[i % len(train_steps)].run(feed_dict={x: batch[0],
                                                                            y_: batch[1],
                                                                            keep_prob1:0.5,
                                                                            keep_prob2:0.5,
                                                                            keep_prob3:0.5})
                '''
            #    train_steps[epoch_number % len(train_steps)].run(feed_dict={x: batch[0].reshape((len(batch[0]),784)),
            #                                                                y_: batch[1],
            #                                                                keep_prob1:0.5,
            #                                                                keep_prob2:0.5})
            #    '''
            else:
                if epoch_iter*cutoff == i:
                    print("Switching to output layer only.")
                train_steps[-1].run(feed_dict={x: batch[0], y_: batch[1],
                                      keep_prob1:0.5, keep_prob2:0.5, keep_prob3:0.5})
                
        np.save('accuracies_gibbs_'+str(cutoff), accuracies)


