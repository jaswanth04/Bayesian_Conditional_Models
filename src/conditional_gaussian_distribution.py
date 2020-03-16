import tensorflow as tf
import numpy as np
from functools import reduce
import json

class GaussianModelBuilder:

    def __init__(self, input_variables, output_variable):
        self.input_variables = input_variables
        self.output_variable = output_variable
        tf.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        self.save_input = {}
        self.save_output = {}

    def build_graph(self, no_of_features):
        # Define placeholders for input and response
        x = tf.placeholder(tf.float64, shape=(None, no_of_features), name='x')
        y = tf.placeholder(tf.float64, shape=(None,), name='y')

        # Define Variables for weight, bias and precision
        w = tf.get_variable(name="w",
                            dtype=tf.float64,
                            initializer=np.random.rand(no_of_features).astype(dtype="float64"))
        b = tf.get_variable(name="b",
                            dtype=tf.float64,
                            initializer=np.float64(np.random.rand()))
        p = tf.get_variable(name="precision",
                            dtype=tf.float64,
                            initializer=np.float64(np.random.rand()).astype(dtype="float64"))

        # Define the Expectation of P(y|x) which is assumed to be a gaussian
        e_y = tf.reduce_sum(tf.multiply(w, x)) + b

        # Define the maximum likelihood function
        log_likelihood = 0.5 * tf.reduce_sum((tf.math.log(p)) - (p * tf.square(y - e_y)))

        expected_sample = tf.random.normal((1, 1), mean=e_y, stddev=1 / tf.sqrt(p), dtype=tf.dtypes.float64,
                                           name="expected_sample")

        return x, y, w, b, p, e_y, log_likelihood, expected_sample

    def train(self, X, Y, batches, no_of_epochs, learning_rate, retrieve_filename=None,
              weights_checkpoint_filename=None):

        no_of_features = X.shape[1]
        x, y, w, b, p, e_y, log_likelihood, expected_sample = self.build_graph(no_of_features)

        # Initialize the variables of retrieve the variables

        if retrieve_filename:
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, retrieve_filename)
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())

        # Define an optimizer
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(-log_likelihood)

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
        self.save_output["prediction"] = expected_sample

    def evaluate(self, base_data):
        x_collection = []
        for data in base_data:
            x_collection.append(reduce(lambda a, b: a + b, [data[key] for key in self.input_variables]))

        X = np.array(x_collection, dtype="float64")

        Y = np.array([data[self.output_variable][0]
                      for data in base_data], dtype="float64")

        no_of_features = X.shape[1]
        x, y, w, b, p, e_y, log_likelihood, expected_sample = self.build_graph(no_of_features)

        z_score = (y - e_y) * tf.sqrt(p)

        return self.sess.run(z_score, feed_dict={x: X, y: Y})

    def save_model(self, filename):
        tf.saved_model.simple_save(self.sess, filename+'/model', self.save_input, self.save_output)
        print("Successfully saved the model in: ", filename)

        with open(filename+'/variables.json','w') as fp:
            json.dump({"input": self.input_variables, "output": self.output_variable, "prediction_type": "Gaussian"},
                      fp)


