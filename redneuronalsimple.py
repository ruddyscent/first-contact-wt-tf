#!/usr/bin/env python3
"""A simple example of a neural network

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
import tensorflow_datasets as tfds


class Model(tf.keras.Model):
    """A simple fully connected layer

    Attributes:
        W: weight matrix
        b: bias
    """
    def __init__(self, input_size: int, num_class: int):
        super(Model, self).__init__()
        self.W = tf.Variable(tf.zeros([input_size, num_class]))
        self.b = tf.Variable(tf.zeros([num_class]))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.softmax(tf.matmul(inputs, self.W) + self.b)


def cross_entropy(model: tf.keras.Model, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """Calculate cross entropy

    Args:
      model:
      inputs:
      targets: ground truth for the input

    Returns:
      cross entropy for the given inputs
    """
    y = model(inputs)
    loss = -tf.reduce_sum(targets * tf.math.log(y))
    return loss

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
        loss_value = cross_entropy(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.b])

def correct_prediction(y: tf.Tensor, yp: tf.Tensor) -> tf.Tensor:
    """Compare prediction and ground truth

    Args:
      y: prediction
      yp: ground truth

    Returns:
      1D Tensor of correctness
    """
    return tf.equal(tf.argmax(y, 1), tf.argmax(yp, 1))

def accuracy(y: tf.Tensor, yp: tf.Tensor) -> float:
    """Compare prediction and ground truth

    Args:
      y: prediction
      yp: ground truth

    Returns:
      The reduced tensor of accuracy
    """
    correctness = correct_prediction(y, yp)
    acc = tf.math.reduce_mean(tf.cast(correctness, "float"))
    return acc

def main(args: argparse.Namespace):
    """Main entry point of the app
    """
    logger = tf.get_logger()
    logger.info("redneuronalsimple")
    logger.info(args)

    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    ds = mnist_builder.as_dataset(split="train")
    ds = ds.repeat().shuffle(1024).batch(args.batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    model = Model(args.size_of_input, args.num_classes)
    optimizer = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9)

    for i, batch in enumerate(ds):
        if i > args.num_iterations:
            break

        x = tf.reshape(batch['image'] / 255, [args.batch_size, args.size_of_input])
        yp = tf.one_hot(batch['label'], args.num_classes)
        grads = grad(model, x, yp)
        optimizer.apply_gradients(zip(grads, [model.W, model.b]))
        print(f"스텝 {i:03d}에서 정확도: {accuracy(model(x), yp):.3f}")

if __name__ == "__main__":
    """This is executed when run from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, action="store", default=100)
    parser.add_argument("-s", "--size_of_input", type=int, action="store", default=28*28)
    parser.add_argument("-c", "--num_classes", type=int, action="store", default=10)
    parser.add_argument("-i", "--num_iterations", type=int, action="store", default=1000)
    args = parser.parse_args()
    main(args)
