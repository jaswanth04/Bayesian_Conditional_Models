import tensorflow as tf
import numpy as np
from functools import reduce
import json


class MultinomialModelBuilder:

    def __init__(self, input_variables, output_variable):
        self.input_variables = input_variables
        self.output_variable = output_variable
        tf.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        self.save_input = {}
        self.save_output = {}

    def build_graph(self, no_of_features, no_of_categories):
        # Define placeholders for input and response
        x = tf.placeholder(tf.float64, shape=(None, no_of_features), name='x')
        y = tf.placeholder(tf.float64, shape=(None, no_of_categories), name='y')

        # Define Variables for weight, bias and precision
        w = tf.get_variable(name="w",
                            dtype=tf.float64,
                            initializer=np.random.rand(no_of_features, no_of_categories).astype(dtype="float64"))
        b = tf.get_variable(name="b",
                            dtype=tf.float64,
                            initializer=np.random.rand(no_of_categories).astype(dtype="float64"))

        # Define the Expectation of P(y|x) which is assumed to be a Multinomial
        y_dist = tf.nn.softmax(tf.matmul(x, w) + b, name="y_dist")

        # Create a constant to adjust log-overflow
        eps = tf.constant(np.array([1e-10] * no_of_categories))

        # Define the maximum likelihood function
        # log_likelihood = tf.reduce_sum(tf.multiply(y, tf.log(y_dist + eps)), axis=[0, 1])
        log_likelihood = tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(y, y_dist), axis=1) + eps), axis=0)

        return x, y, w, b, y_dist, log_likelihood

    def train(self, X, Y, batches, no_of_epochs, learning_rate, retrieve_filename=None,
              weights_checkpoint_filename=None):

        no_of_features = X.shape[1]
        no_of_categories = Y.shape[1]
        x, y, w, b, y_dist, log_likelihood = self.build_graph(no_of_features, no_of_categories)

        prediction = tf.random.categorical(y_dist, 1, name="prediction")

        if retrieve_filename:
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, retrieve_filename)
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())

        # Define an optimizer
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(-log_likelihood)

        # Start the session and initialize the variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(no_of_epochs):
            for batch in batches:
                # print("batch values", batch)
                self.sess.run(train_op, feed_dict={x: X[batch[0]:batch[1]], y: Y[batch[0]:batch[1]]})

            if i % 10 == 0:
                print("Epoch ", i, ": ", self.sess.run(log_likelihood, feed_dict={x: X, y: Y}))

        if weights_checkpoint_filename:
            var_saver = tf.train.Saver()
            var_saver.save(self.sess, weights_checkpoint_filename)

        self.save_input["features_placeholder"] = x
        self.save_output["prediction"] = prediction

    def save_model(self, filename):
        tf.saved_model.simple_save(self.sess, filename + '/model', self.save_input, self.save_output)
        print("Successfully saved the model in: ", filename)

        with open(filename + '/variables.json', 'w') as fp:
            json.dump({"input": self.input_variables, "output": self.output_variable, "prediction_type": "Multinomial"},
                      fp)

