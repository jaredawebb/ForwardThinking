import tensorflow as tf
import numpy as np

class neural_net():
    
    def weight_variable(self, shape):
        # Weight variable helper function
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # Bias variable helper function
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __init__(self, hidden_nodes, input_size):
        
        # For storing the weights/biases as they are learned and frozen
        self.weights = []
        # For tracking validation accuracy at each epoch.
        self.epoch_acc = []

        # Placeholders for input and labels
        self.x = tf.placeholder(tf.float32, shape=[None, input_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        # First layer
        self.W_fc1 = self.weight_variable([input_size, hidden_nodes])
        self.b_fc1 = self.bias_variable([hidden_nodes])
        
        # First activations/hidden layer
        self.h_fc1 = tf.nn.relu(tf.matmul(self.x, self.W_fc1) + self.b_fc1)
        
        # Output layer for training.  Discarded once the layer is trained.
        self.W_fc2 = self.weight_variable([hidden_nodes, 10])
        self.b_fc2 = self.bias_variable([10])
        
        self.y = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        
        
    def train_first_layer(self, sess, mnist, iterations=5000, batch_size=50):

        # Training the first layer
        
        # Standard tensorflow setup - define loss, choose an optimizer, and define
        # a training step.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        optimizer = tf.train.AdamOptimizer(1e-4)

        train_step = optimizer.minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # Train the model
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            
            # Get the next batch for training
            batch = mnist.train.next_batch(batch_size)
            
            # When an epoch finishes print the models validation accuracy and store it
            if (i * batch_size) % len(mnist.train.images) == 0 and i > 0:
                print("Epoch Complete")
                self.epoch_acc.append(accuracy.eval(feed_dict={self.x:mnist.test.images, self.y_:mnist.test.labels}))
            
            # Print training accuracy everything 1000 iterations.
            if i%1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        self.x:batch[0], self.y_: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})

        # Validation accuracy after training
        acc = accuracy.eval(feed_dict={self.x:mnist.test.images, self.y_:mnist.test.labels})
        
        # Ignore this stuff - obsolete but I don't want to get rid of it yet.
        activations = self.h_fc1.eval(feed_dict={self.x: mnist.train.images, self.y_:mnist.train.labels})
        activations_test = self.h_fc1.eval(feed_dict={self.x: mnist.test.images, self.y_:mnist.test.labels})
        
        # Store the weights that we have trained
        self.weights.append((self.W_fc1.eval(), self.b_fc1.eval()))
        
        print("First Layer Accuracy %f" % acc)
        
        return activations, activations_test, acc, self.weights
    
    def train_next_layer(self, sess, mnist, H, H_test, hidden_nodes=64, iterations=10000, batch_size=50):
        
        def next_batch(A, i, j):
            return A[i:j,:]
        
        input_size = H.shape[1]
        print(input_size)
             
        # Same as before
        x = tf.placeholder(tf.float32, shape=[None, input_size])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        W_fc1 = self.weight_variable([input_size, hidden_nodes])
        b_fc1 = self.bias_variable([hidden_nodes])
        
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        
        W_fc2 = self.weight_variable([hidden_nodes, 10])
        b_fc2 = self.bias_variable([10])
        
        y = tf.matmul(h_fc1, W_fc2) + b_fc2
                         
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        optimizer = tf.train.AdamOptimizer(1e-4)

        train_step = optimizer.minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        sess.run(tf.global_variables_initializer())
        index = 0
        size = len(H)
        for i in range(iterations):
            batch = mnist.train.next_batch(batch_size)
            # Retrieve the batch numpy array
            curr_h = batch[0].copy()
            
            # Pass the batch through the previously trained, frozen layers.
            # This becomes the "training data" for the next layer
            for j in range(len(self.weights)):
                curr_h = np.dot(curr_h, self.weights[j][0]) + self.weights[j][1]
            
            # Epoch printage/storage
            if (i * batch_size) % len(mnist.train.images) == 0 and i > 0:
                print("Epoch Complete")
                test_h = mnist.test.images.copy()
                
                for j in range(len(self.weights)):
                    test_h = np.dot(test_h, self.weights[j][0]) + self.weights[j][1]
                
                self.epoch_acc.append(accuracy.eval(feed_dict={x:test_h, y_:mnist.test.labels}))
            
            # Training printage/storage
            if i%1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x:curr_h, y_: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x:curr_h, y_: batch[1]})

        # Store our new weights
        self.weights.append((W_fc1.eval(), b_fc1.eval()))

        # Final accuracy after training the layer
        acc = accuracy.eval(feed_dict={x:H_test, y_:mnist.test.labels})
        activations = h_fc1.eval(feed_dict={x: H, y_:mnist.train.labels})
        activations_test = h_fc1.eval(feed_dict={x: H_test, y_:mnist.test.labels})
        
        print("Next Layer Accuracy %f" % acc)
        
        return activations, activations_test, acc, self.weights
    
    def train_output_layer(self, sess, mnist, H, H_test, weights, iterations=10000, batch_size=50):
        
        # This can be done with train_next_layer, but for historical reasons it is still here.
        # The only difference is that we set up the previously learned weights as tensorflow
        # constants, and the train the output layer.
        
        def next_batch(A, i, j):
            return A[i:j,:]
        
        input_size = 28*28
        print(input_size)
        
        x = tf.placeholder(tf.float32, shape=[None, input_size])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        W_fc1 = tf.constant(weights[0][0], shape=weights[0][0].shape)
        b_fc1 = tf.constant(weights[0][1], shape=weights[0][1].shape)

        hidden_activ = []
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        hidden_activ.append(h_fc1)
        
        for i in range(1,len(weights)-1):
            curr_W = tf.constant(weights[i][0], shape=weights[i][0].shape)
            curr_b = tf.constant(weights[i][1], shape=weights[i][1].shape)
            curr_h = tf.nn.relu(tf.matmul(hidden_activ[i-1], curr_W) + curr_b)
            hidden_activ.append(curr_h)
            

        W_fc2 = self.weight_variable([weights[-1][0].shape[0], 10])
        b_fc2 = self.bias_variable([10])

        y = tf.matmul(hidden_activ[-1], W_fc2) + b_fc2
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        optimizer = tf.train.AdamOptimizer(1e-4)

        train_step = optimizer.minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        #index = 0
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            batch = mnist.train.next_batch(batch_size)

            if (i * batch_size) % len(mnist.train.images) == 0 and i > 0:
                print("Epoch Complete")
                self.epoch_acc.append(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
            
            if i%1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x:batch[0], y_: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x:batch[0], y_: batch[1]})
        self.weights.append((W_fc1.eval(), b_fc1.eval()))
        
        acc = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
        activations = h_fc1.eval(feed_dict={x: mnist.train.images, y_:mnist.train.labels})
        activations_test = h_fc1.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
        
        print("Final Accuracy %f" % acc)
        
            
        
    
        
        
        
        
        