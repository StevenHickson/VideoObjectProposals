#!/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import kmeans
from tensorflow.python.framework import constant_op
import sys
from city_scape_info import TrainIdToName
from city_scape_info import OriginalIdToName
from city_scape_info import ImportantLabelMapping
from city_scape_info import PurityLabelMapping
from city_scape_info import PurityLabelMapping2
from city_scape_info import PurityLabelMapping3
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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


def purity_score(classes, clusters):
    clusters = np.array(clusters)
    classes = np.array(classes)
    A = np.c_[(clusters,classes)]

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]


def process_clusters(name, labels, predictions):
    cluster_counts = dict()
    for l,p in zip(labels,predictions):
        if l not in cluster_counts:
            cluster_counts[int(l)] = [0] * FLAGS.num_k
        cluster_counts[int(l)][p] += 1

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

    newLabels = []
    newPreds = []
    purity_labels = []
    purity_preds = []
    purity_labels2 = []
    purity_preds2 = []
    purity_labels3 = []
    purity_preds3 = []
    for l,p in zip(labels, predictions):
        new_l = ImportantLabelMapping(l)
        if new_l >= 0:
            newLabels.append(new_l)
            newPreds.append(p)
        new_l = PurityLabelMapping(l)
        if new_l >= 0:
            purity_labels.append(new_l)
            purity_preds.append(p)
        new_l = PurityLabelMapping2(l)
        if new_l >= 0:
            purity_labels2.append(new_l)
            purity_preds2.append(p)
        new_l = PurityLabelMapping3(l)
        if new_l >= 0:
            purity_labels3.append(new_l)
            purity_preds3.append(p)

    print('\nNormalized Mutual Information Score')
    print normalized_mutual_info_score(newLabels,newPreds)

    print('\nAdjusted Mutual Information Score')
    print adjusted_mutual_info_score(newLabels,newPreds)

    print('\nPurity Score')
    score = purity_score(purity_labels,purity_preds)
    print score

    print('\nPurity Score2')
    score = purity_score(purity_labels2,purity_preds2)
    print score

    print('\nPurity Score3')
    score = purity_score(purity_labels3,purity_preds3)
    print score


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
