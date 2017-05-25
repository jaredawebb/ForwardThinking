# This won't work with interactive session.  Need to pass the whole session through.

import tensorflow as tf
import functools
import numpy as np

class neural_net():
    
    def get_tensor_size(self, tensor):
        
        from operator import mul
        return functools.reduce(mul, (d.value for d in tensor.get_shape()), 1)

    def get_hessian(self, grads):
        
        hessian = []
        for grad, var in grads:
            
            if grad is None:
                grad2 = 0
            else:
                grad = 0 if None else grad
                grad2 = tf.gradients(grad, var)
                grad2 = 0 if None else grad2

            hessian.append(grad2)
    
        return hessian
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def return_slice(self, params, loc, size, shape):
        return tf.reshape(tf.slice(params, begin=[loc, 0], size=size), shape=shape)

    def __init__(self, hidden_nodes=[128,128,128,128], weights=None, biases=None):

        tf.reset_default_graph()
        num_layers = len(hidden_nodes)
        self.hidden_weights = []
        self.hidden_bias = []
        self.hidden_activ = []
        self.epoch_acc = []
        
        if weights is None:

            self.x = tf.placeholder(tf.float32, shape=[None, 28*28])
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
            
            
            # Grab slices from the parameter list
            W_fc1 = self.weight_variable([28*28, hidden_nodes[0]])
            #W_fc1 = self.return_slice(self.parameters, loc, [28*28*hidden_nodes[0], 1], [28*28, hidden_nodes[0]])
            
            b_fc1 = self.bias_variable([hidden_nodes[0]])
                                                                                    
            h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)

            self.hidden_weights.append(W_fc1)
            self.hidden_bias.append(b_fc1)
            self.hidden_activ.append(h_fc1)

            for i in range(1,num_layers):
                curr_W = self.weight_variable([hidden_nodes[i-1], hidden_nodes[i]])
                curr_b = self.bias_variable([hidden_nodes[i-1]])
                curr_h = tf.nn.relu(tf.matmul(self.hidden_activ[i-1], curr_W) + curr_b)

                self.hidden_weights.append(curr_W)
                self.hidden_bias.append(curr_b)
                self.hidden_activ.append(curr_h)

            W_fc2 = self.weight_variable([hidden_nodes[-1], 10])
            b_fc2 = self.bias_variable([10])

            self.hidden_weights.append(W_fc2)
            self.hidden_bias.append(b_fc2)

            self.y = tf.matmul(self.hidden_activ[-1], W_fc2) + b_fc2
            
        elif weights is not None and biases is None:
            
            self.x = tf.placeholder(tf.float32, shape=[None, 28*28])
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

            W_fc1 = tf.constant(weights[0], shape=weights[0].shape)
            b_fc1 = self.bias_variable([hidden_nodes[0]])

            h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)
            
            self.hidden_weights.append(W_fc1)
            self.hidden_bias.append(b_fc1)
            self.hidden_activ.append(h_fc1)

            for i in range(1,num_layers):
                curr_W = tf.constant(weights[i], shape=weights[i].shape)
                curr_b = self.bias_variable([hidden_nodes[i]])
                curr_h = tf.nn.relu(tf.matmul(self.hidden_activ[i-1], curr_W) + curr_b)

                self.hidden_weights.append(curr_W)
                self.hidden_bias.append(curr_b)
                self.hidden_activ.append(curr_h)

            W_fc2 = tf.constant(weights[-1], shape=weights[-1].shape)
            b_fc2 = self.bias_variable([10])

            self.hidden_weights.append(W_fc2)
            self.hidden_bias.append(b_fc2)

            self.y = tf.matmul(self.hidden_activ[-1], W_fc2) + b_fc2
            
        else:
            
            self.x = tf.placeholder(tf.float32, shape=[None, 28*28])
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

            W_fc1 = tf.constant(weights[0], shape=weights[0].shape)
            b_fc1 = tf.constant(biases[0], shape=biases[0].shape)

            h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)
            
            self.hidden_weights.append(W_fc1)
            self.hidden_bias.append(b_fc1)
            self.hidden_activ.append(h_fc1)

            for i in range(1,num_layers):
                curr_W = tf.constant(weights[i], shape=weights[i].shape)
                curr_b = tf.constant(biases[i], shape=biases[i].shape)
                curr_h = tf.nn.relu(tf.matmul(self.hidden_activ[i-1], curr_W) + curr_b)

                self.hidden_weights.append(curr_W)
                self.hidden_bias.append(curr_b)
                self.hidden_activ.append(curr_h)

            W_fc2 = tf.constant(weights[-1], shape=weights[-1].shape)
            b_fc2 = tf.constant(biases[-1], shape=biases[-1].shape)

            self.hidden_weights.append(W_fc2)
            self.hidden_bias.append(b_fc2)

            self.y = tf.matmul(self.hidden_activ[-1], W_fc2) + b_fc2


    def train_network(self, sess, mnist, iterations=5000, batch_size=50, mile_marker=1000):

        print("\tSetting up training op")
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_step = optimizer.minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        print("\tInitializing Variables")
        
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            
            if (i * batch_size) % len(mnist.train.images) == 0 and i > 0:
                print("Epoch Complete")
                self.epoch_acc.append(accuracy.eval(feed_dict={self.x:mnist.test.images, self.y_:mnist.test.labels}))
            
            batch = mnist.train.next_batch(batch_size)
            if i % mile_marker == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    self.x:batch[0], self.y_: batch[1]})
                print("\tstep %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
        
        res = accuracy.eval(feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})
        
        print("\ttest accuracy %g" % res)

    
    def run_network(self, sess, mnist):
        
        # This is for networks that are already trained.
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())

        res = accuracy.eval(feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels})
        
        print("\ttest accuracy %g" % res)
        
        
        
        return res
    
    def optimal_brain_damage(self, sess, hess, mnist, batch_size=50):
        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #train_vars = tf.trainable_variables()
        
        sess.run(tf.global_variables_initializer())
        
        train_vars_copy = sess.run([tf.identity(var) for var in self.train_vars])
        
        k = tf.placeholder(tf.int32)

        def scatter_update(saliency, variables):
            
            shape = self.get_tensor_size(variables)  # Get the number of variables
            
            # Find the values and indices of the top k% of the variables
            values, indices = tf.nn.top_k(-1 * saliency**2, tf.cast(k * shape / 100, tf.int32)) 
            data_subset = tf.gather(variables, indices)
            updated_data_subset = 0*data_subset
            
            # Update variables tensor at indices with zeros
            return tf.scatter_update(variables, indices, updated_data_subset)

        def scatter_restore(saliency, variables1, variables2):
            shape = self.get_tensor_size(variables2)
            values, indices = tf.nn.top_k(-1 * saliency**2, tf.cast(k * shape / 100, tf.int32))
            values = tf.gather(variables1, indices)
            return tf.scatter_update(variables2, indices, values)

        scatter_update_op = [scatter_update(sal, var) for sal, var in zip([hess], self.train_vars)]
        scatter_restore_op = [scatter_restore(sal, var1, var2) for sal, var1, var2 in
                              zip([hess], train_vars_copy, self.train_vars)]

        for count in range(1, 20):
            batch = mnist.train.next_batch(batch_size)
            print(count)
            feed_dict={self.x: batch[0], self.y_: batch[1], k:count}
            #A = hess.eval(feed_dict=feed_dict)
            #print(A)
            #print(np.count_nonzero(A))
            
            print(np.count_nonzero(self.train_vars[0].eval()))
            sess.run(scatter_update_op, feed_dict=feed_dict)
            print(np.count_nonzero(self.train_vars[0].eval()))
            print('Variables Percent: %d, Test accuracy: %g' % ((100 - count), accuracy.eval(feed_dict={
                self.x: mnist.test.images, self.y_: mnist.test.labels})))
            sess.run(scatter_restore_op, feed_dict=feed_dict)
        
        
