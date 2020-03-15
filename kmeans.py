from __future__ import absolute_import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

NUM_POINTS = 2000
K = 4

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        vector_a = tf.random.normal((NUM_POINTS // 2, 2), mean=0, stddev=0.9)
        vector_b = tf.random.normal((NUM_POINTS // 2, 2), mean=[3, 1], stddev=0.5)
        self.vectors = tf.random.shuffle(tf.concat([vector_a, vector_b], 0))
        self.centroids = self.vectors[:K, :]
        self.assignment = tf.random.uniform([NUM_POINTS], minval=0, maxval=K, dtype=tf.int32)

    def update_centroids(self):
        centroids = []
        for c in range(K):
            pts_idx = tf.reshape(tf.where(tf.equal(self.assignment, c)), [1, -1])
            pts = tf.gather(self.vectors, pts_idx)
            centroids.append(tf.reduce_mean(pts, 1))
        self.centroids = tf.reshape(tf.convert_to_tensor(centroids, dtype=tf.float32), [K, 2])
        expanded_vectors = tf.expand_dims(self.vectors, 0)
        expanded_centroides = tf.expand_dims(self.centroids, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2)
        self.assignment = tf.argmin(distances, 0)

model = Model()
for i in range(100):
    model.update_centroids()

df = pd.DataFrame({"x": model.vectors[:, 0], "y": model.vectors[:, 1], "cluster": model.assignment.numpy()})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster")
plt.show()
