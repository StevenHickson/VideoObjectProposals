#!/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import kmeans
from tensorflow.python.framework import constant_op
import sys
from city_scape_info import TrainIdToName
from city_scape_info import OriginalIdToName
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

# Input is the train set, the validation set, # of clusters
# An example is python test_kmeans.py /data/gt_proposals/embeddings_baseline_train.npy /data/gt_proposals/embeddings_baseline.npy

train_set = np.load(sys.argv[1])
val_set = np.load(sys.argv[2])
num_k = int(sys.argv[3])

clusterer = kmeans.KMeansClustering(
      num_clusters=num_k,
      use_mini_batch=False)
x = train_set
labels = x[:,0:1]
ks = x[:1:2]
x = x[:,2:]

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
ks_val = x_val[:,1:2]
x_val = x_val[:,2:]
predictions = np.array(list(clusterer.predict_cluster_idx(
        input_fn=_input_fn(x_val.astype(np.float32), num_epochs=1))))

cluster_counts = dict()
for l,p in zip(labels_val.flatten(),predictions):
    if l not in cluster_counts:
        cluster_counts[l] = [0] * num_k
    cluster_counts[l][p] += 1

cluster_counts_ete = dict()
for l,p in zip(labels_val.flatten(),ks_val.flatten()):
    if l not in cluster_counts_ete:
        cluster_counts_ete[l] = [0] * num_k
    cluster_counts_ete[l][int(p)] += 1

#print(cluster_counts)
minVal = min(cluster_counts)
maxVal = max(cluster_counts)
if minVal < 11 or maxVal > 18:
    origId = True
else:
    origId = False
print('\n----------K Cluster results----------\n\n')
for k,v in cluster_counts.iteritems():
    if origId:
        print(OriginalIdToName[k] + ': ' + str(v))
    else:
        print(TrainIdToName[k] + ': ' + str(v))

print('\n----------K end-to-end results----------\n\n')
for k,v in cluster_counts_ete.iteritems():
    if origId:
        print(OriginalIdToName[k] + ': ' + str(v))
    else:
        print(TrainIdToName[k] + ': ' + str(v))

print('\nNormalized Mutual Information Score')
print normalized_mutual_info_score(labels_val.flatten(),predictions)

print('\nAdjusted Mutual Information Score')
print adjusted_mutual_info_score(labels_val.flatten(),predictions)

