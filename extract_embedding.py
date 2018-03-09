#!/bin/python

import numpy as np
import tensorflow as tf
import sys

# Args are: checkpoint path, text file, output file.
# An exampleis python extract_embedding.py /data/model/inception/classify_image_graph_def.pb /data/proposals_train/info.txt /data/proposals/embeddings_baseline_train.npy

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(sys.argv[1], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

embedding_list = []
labels_list = []
#sess = tf.Session('local', graph=tf.get_default_graph())
#with sess, sess.graph.device(tf.ReplicaDeviceSetter(ps_tasks=0, worker_device='/gpu:0')):
# Creates graph from saved GraphDef.
create_graph()
c = 0
gpu_options = tf.GPUOptions(allow_growth = True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    embedding_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for line in open(sys.argv[2]):
        fields = line.split(',')
        if int(fields[3]) == 1:
            image_data = tf.gfile.FastGFile(fields[0], 'rb').read()
            embeddings = sess.run(embedding_tensor, {'DecodeJpeg/contents:0': image_data})
            labels_list.append(int(fields[2]))
            embedding_list.append(embeddings)
            c += 1

labels_list = np.array(labels_list).reshape(c, 1)
embedding_list = np.array(embedding_list)
embedding_list = embedding_list.reshape(c, 2048)

save_array = np.hstack((labels_list, embedding_list))
np.save(sys.argv[3], save_array)
