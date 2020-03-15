from __future__ import absolute_import

import matplotlib.pyplot as plt

import tensorflow as tf

NUM_POINTS = 1000

x1 = tf.random.normal([NUM_POINTS], mean=0, stddev=0.55)
noise = tf.random.normal([NUM_POINTS], mean=0, stddev=0.03)
y1 = x1 * 0.3 + 0.1 + noise

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(tf.random.uniform([1], -1, 1), name='weight')
        self.b = tf.Variable(tf.zeros([1]), name='bias')

    def call(self, inputs):
        return inputs * self.W + self.b


def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.b])

model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

for i in range(8):
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
