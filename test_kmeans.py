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

tf.app.flags.DEFINE_string(
    'train_file', '', 'The train_embeddings to use.')

tf.app.flags.DEFINE_string(
    'test_file', '', 'The test_embeddings to use.')

tf.app.flags.DEFINE_integer(
    'num_k', 3, 'The number of k clusters to use.')

tf.app.flags.DEFINE_bool(
    'endtoend', False, 'Whether to use endtoend k clusters.')

FLAGS = tf.app.flags.FLAGS

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

def process_clusters(name, labels, predictions):
    cluster_counts = dict()
    for l,p in zip(labels,predictions):
        if l not in cluster_counts:
            cluster_counts[l] = [0] * FLAGS.num_k
        cluster_counts[l][p] += 1

    minVal = min(cluster_counts)
    maxVal = max(cluster_counts)
    if minVal < 11 or maxVal > 18:
        origId = True
    else:
        origId = False
    print('\n----------' + name + ' results----------\n\n')
    for k,v in cluster_counts.iteritems():
        if origId:
            print(OriginalIdToName[k] + ': ' + str(v))
        else:
            print(TrainIdToName[k] + ': ' + str(v))

    print('\nNormalized Mutual Information Score')
    print normalized_mutual_info_score(labels,predictions)

    print('\nAdjusted Mutual Information Score')
    print adjusted_mutual_info_score(labels,predictions)

def main(_):
    train_set = np.load(FLAGS.train_file)
    val_set = np.load(FLAGS.test_file)
    num_k = int(FLAGS.num_k)

    clusterer = kmeans.KMeansClustering(
          num_clusters=num_k,
          use_mini_batch=False)
    labels = train_set[:,0:1]
    if FLAGS.endtoend:
        ks = train_set[:1:2]
        x = train_set[:,2:]
    else:
        x = train_set[:,1:]

    clusterer.fit(
          input_fn=_input_fn(x.astype(np.float32)), steps=10)

    labels_val = val_set[:,0:1]
    if FLAGS.endtoend:
        ks_val = val_set[:,1:2]
        x_val = val_set[:,2:]
    else:
        x_val = val_set[:,1:]
    predictions = np.array(list(clusterer.predict_cluster_idx(
            input_fn=_input_fn(x_val.astype(np.float32), num_epochs=1))))

    process_clusters('K Cluster', labels_val.flatten(), predictions)

    if FLAGS.endtoend:
        process_clusters('K end-to-end', labels_val.flatten(), ks_val.flatten().astype(int))


if __name__ == '__main__':
    tf.app.run()
