from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.keras.backend.clear_session()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.b

def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.b])

model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

for i in range(1000):
    batch_xs, batch_ys = mnist
    grads = grad(model, x1, y1)
    optimizer.apply_gradients(zip(grads, [model.W, model.b]))
    print(f"스텝 {i:03d}에서 손실: {loss(model, x1, y1):.3f}")
