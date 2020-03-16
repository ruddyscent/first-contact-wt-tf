#!/usr/bin/env python3
"""An example of k-means clustering

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf


class Model(tf.keras.Model):
    """An implementation of k-means clustering

    Attributes:
        vectors: a set of observations
        centroids: means for observations assigned to each cluster
        assignment: labels of each observations
    """
    def __init__(self, num_points: int, num_clusters: int):
        super(Model, self).__init__()
        vector_a = tf.random.normal((num_points // 2, 2), mean=0, stddev=0.9)
        vector_b = tf.random.normal((num_points // 2, 2), mean=[3, 1], stddev=0.5)
        self.vectors = tf.random.shuffle(tf.concat([vector_a, vector_b], 0))
        self.centroids = self.vectors[:num_clusters, :]
        self.assignment = tf.random.uniform([num_points], minval=0, maxval=num_clusters, 
                                            dtype=tf.int32)
        self.num_points = num_points
        self.num_clusters = num_clusters

    def update_centroids(self):
        """Calculate a new centroids and assign vectors to the new centroids
        """
        centroids = []
        for c in range(self.num_clusters):
            pts_idx = tf.reshape(tf.where(tf.equal(self.assignment, c)), [1, -1])
            pts = tf.gather(self.vectors, pts_idx)
            centroids.append(tf.reduce_mean(pts, 1))
        centroids = tf.convert_to_tensor(centroids, dtype=tf.float32)
        self.centroids = tf.reshape(centroids, [self.num_clusters, 2])

    def update_labels(self):
        """Update the labels of each observations
        """
        expanded_vectors = tf.expand_dims(self.vectors, 0)
        expanded_centroides = tf.expand_dims(self.centroids, 1)
        difference = tf.subtract(expanded_vectors, expanded_centroides)
        distances = tf.reduce_sum(tf.square(difference), 2)
        self.assignment = tf.argmin(distances, 0)


def main(args: argparse.Namespace):
    """Main entry point of the app
    """
    logger = tf.get_logger()
    logger.info("kmeans")
    logger.info(args)

    model = Model(args.num_points, args.num_clusters)
    for _ in range(args.num_iteration):
        model.update_centroids()
        model.update_labels()

    df = pd.DataFrame({"x": model.vectors[:, 0],
                       "y": model.vectors[:, 1],
                       "cluster": model.assignment.numpy()})
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster")
    plt.show()

if __name__ == "__main__":
    """This is executed when run from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_points", type=int, action="store", default=2000)
    parser.add_argument("-k", "--num_clusters", type=int, action="store", default=4)
    parser.add_argument("-i", "--num_iteration", type=int, action="store", default=100)
    args = parser.parse_args()
    main(args)
