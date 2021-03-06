import tensorflow as tf
import numpy as np
import xgboost

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=7,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    zoom_range=.1)

x_train = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
x_test = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)

datagen.fit(x_train)
images = datagen.flow(x_train, mnist.train.labels, batch_size=55000)

epochs = 10

images_train = np.zeros((55000*epochs, 784))
images_labels = np.zeros((55000*epochs, 10))
for i in range(epochs):
    batch = images.next()
    images_train[i*55000:(i+1)*55000, :] = batch[0].reshape((55000, 784))
    images_labels[i*55000:(i+1)*55000] = batch[1]
    
xgb = xgboost.XGBClassifier(objective='multi:softmax')
xgb.fit(images_train, np.argmax(images_labels, axis=1))

yhat = xgb.predict(mnist.test.images)
accuracy = (yhat == np.argmax(mnist.test.labels, axis=1))
accuracy = sum(accuracy)/len(accuracy)
np.save('prediction', yhat)
np.save('accuracy', accuracy)
