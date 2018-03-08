#!/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import kmeans
from tensorflow.python.framework import constant_op
import sys

# Input is the train set and the validation set
# An example is python test_kmeans.py /data/gt_proposals/embeddings_baseline_train.npy /data/gt_proposals/embeddings_baseline.npy

train_set = np.load(sys.argv[1])
val_set = np.load(sys.argv[2])

clusterer = kmeans.KMeansClustering(
      num_clusters=2,
      use_mini_batch=False)
x = train_set
labels = x[:,0:1]
x = x[:,1:]

def _input_fn(tensor, num_epochs=None):
    """Constructs an input_fn for KMeansClustering estimator.

    Args:
      tensor: A 2D numpy.ndarray of data points.
      num_epochs: A positive integer (optional).  If specified, limits the
        number of steps the output tensor may be evaluated.

    Returns:
      An input_fn.
    """
    def _constant_input_fn():
          return (
          tf.train.limit_epochs(tf.constant(tensor), num_epochs=num_epochs),
          None)
    return _constant_input_fn

clusterer.fit(
      input_fn=_input_fn(x.astype(np.float32)), steps=10)

x_val = val_set
labels_val = x_val[:,0:1]
x_val = x_val[:,1:]
predictions = np.array(list(clusterer.predict_cluster_idx(
        input_fn=_input_fn(x_val.astype(np.float32), num_epochs=1))))

cluster_counts = dict()
for l,p in zip(labels_val.flatten(),predictions):
    if l not in cluster_counts:
        cluster_counts[l] = [0,0,0]
    cluster_counts[l][p] += 1

print(cluster_counts)
