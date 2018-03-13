#!/bin/python

import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.contrib.learn.python.learn.estimators import kmeans
from tensorflow.python.framework import constant_op

tf.app.flags.DEFINE_string(
    'file_list', '', 'The list of files to use.')

tf.app.flags.DEFINE_string(
    'train_file', '', 'The train_embeddings to use.')

tf.app.flags.DEFINE_string(
    'test_file', '', 'The test_embeddings to use.')

tf.app.flags.DEFINE_string(
    'output_folder', '', 'The output folder to use.')

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

def main(_):
    filelist = []
    for line in open(FLAGS.file_list):
        filelist.append(line)

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

    file_dict = dict()
    c = 0
    for f in filelist:
        f = f.strip()
        fields = f.split(',')
        if int(fields[3]) == 1:
            # See if field[1] exists in dict
            key = fields[1]
            info = fields[0] + ',' + str(predictions[c]) + ',' + fields[4] + ',' + fields[5] + ',' + fields[6] + ',' + fields[7]
            if key not in file_dict:
                file_dict[key] = []
            file_dict[key].append(info)
            c += 1

    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    for k,v in file_dict.iteritems():
        fields = k.strip().split('/')
        filename = os.path.join(FLAGS.output_folder,fields[-1].replace('leftImg8bit','results'))
        img = cv2.imread(k.strip())
        for info in v:    
		fields = info.split(',')
		label = int(fields[1])
    		x = int(fields[2])
    		y = int(fields[3])
    		w = int(fields[4])
    		h = int(fields[5])
    		cv2.rectangle(img,(x,y),(x+w,y+h),colors[label],2)
        cv2.imwrite(filename, img)
if __name__ == '__main__':
    tf.app.run()
