#!/usr/bin/env python3
"""A simple example of a convolutional network

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
    def __init__(self, input_size: int, num_class: int, keep_prob: int):
        super(Model, self).__init__()
        self.W_conv1 = Model.weight_variable([5, 5, 1, 32])
        self.b_conv1 = Model.bias_variable([32])

        self.W_conv2 = Model.weight_variable([5, 5, 32, 64])
        self.b_conv2 = Model.bias_variable([64])

        self.W_fc1 = Model.weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = Model.bias_variable([1024])

        self.W_fc2 = Model.weight_variable([1024, 10])
        self.b_fc2 = Model.bias_variable([10])

    def call(self, inputs: tf.Tensor, keep_prob: float=1.0) -> tf.Tensor:
        h_conv1 = tf.nn.relu(Model.conv2d(inputs, self.W_conv1) + self.b_conv1)
        h_pool1 = Model.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(Model.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = Model.max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        if keep_prob == 1:
            h_fc1_drop = h_fc1
        else:
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2)
        return y_conv
        
    @classmethod
    def weight_variable(cls, shape):
        initial = tf.random.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @classmethod
    def bias_variable(cls, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @classmethod
    def conv2d(cls, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    @classmethod
    def max_pool_2x2(cls, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cross_entropy(model: tf.keras.Model, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """Calculate cross entropy

    Args:
      model:
      inputs:
      targets: ground truth for the input

    Returns:
      cross entropy for the given inputs
    """
    y = model(inputs, 0.5)
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
    return tape.gradient(loss_value, [model.W_conv1, model.b_conv1, model.W_conv2, model.b_conv2, model.W_fc1,  model.b_fc1,  model.W_fc2,  model.b_fc2])

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
    logger.info("cnn")
    logger.info(args)

    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    ds = mnist_builder.as_dataset(split="train")
    ds = ds.repeat().shuffle(1024).batch(args.batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    model = Model(args.size_of_input, args.num_classes, 1.0)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    keep_prob = 0.5
    for i, batch in enumerate(ds):
        if i > args.num_iterations:
            break

        x = batch['image'] / 255
        yp = tf.one_hot(batch['label'], args.num_classes)
        grads = grad(model, x, yp)
        optimizer.apply_gradients(zip(grads, [model.W_conv1, model.b_conv1, model.W_conv2, model.b_conv2, model.W_fc1,  model.b_fc1,  model.W_fc2,  model.b_fc2]))
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
