#!/usr/bin/env python3
"""A simple example of a linear regression

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import matplotlib.pyplot as plt
import tensorflow as tf


class Model(tf.keras.Model):
    """A simple fully connected layer

    Attributes:
        W: weight matrix
        b: bias
    """
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(tf.random.uniform([1], -1, 1), name='weight')
        self.b = tf.Variable(tf.zeros([1]), name='bias')
        print(type(self.W))
        print(type(self.b))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calculate a logit

        Args:
          inputs:

        Returns:
          logit of this model
        """
        return inputs * self.W + self.b


def loss(model: tf.keras.Model, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """Calculate mean square error

    Args:
      model:
      inputs:
      targets: ground truth for the input

    Returns:
      mse for the given inputs
    """
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model: tf.keras.Model, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """Compute the derivative of loss with respect to the weight and bias.

    Args:
      model:
      inputs:
      targets:

    Returns:
      Gradients of the output with respect to intermediate values.
    """
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.b])

def main(args: argparse.Namespace):
    """Main entry point of the app
    """
    logger = tf.get_logger()
    logger.info("regression")
    logger.info(args)

    x1 = tf.random.normal([args.num_points], mean=0, stddev=0.55)
    noise = tf.random.normal([args.num_points], mean=0, stddev=0.03)
    y1 = x1 * 0.3 + 0.1 + noise

    model = Model()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

    for i in range(args.num_iteration):
        grads = grad(model, x1, y1)
        optimizer.apply_gradients(zip(grads, [model.W, model.b]))
        print(f"스텝 {i:03d}에서 손실: {loss(model, x1, y1):.3f}")

    print(f"W = {model.W.numpy()}, b = {model.b.numpy()}")
    plt.plot(x1, y1, 'ro', label='data')
    plt.plot(x1, model.W * x1 + model.b, label='regression')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    """This is executed when run from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_points", type=int, action="store", default=1000)
    parser.add_argument("-i", "--num_iteration", type=int, action="store", default=8)
    args = parser.parse_args()
    main(args)
