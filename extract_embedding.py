#!/bin/python

import numpy as np
import tensorflow as tf
import cv2

# Args are: checkpoint path, text file, output file, and embedding layer name.
# An exampleis python extract_embedding.py /data/model/inception/classify_image_graph_def.pb /data/proposals_train/info.txt /data/proposals/embeddings_baseline_train.npy extra_bottleneck_ops/Wx_plus_b/add:0
# Another embedding layer might be: pool_3:0

tf.app.flags.DEFINE_string(
    'checkpoint', '', 'The checkpoint to use.')

tf.app.flags.DEFINE_string(
    'filelist', '', 'The filelist to use.')

tf.app.flags.DEFINE_string(
    'save_file', '', 'The output file to save to.')

tf.app.flags.DEFINE_string(
    'input_tensor', 'DecodeJpeg/contents:0', 'The input tensorf name to use.')

tf.app.flags.DEFINE_string(
    'output_tensor', 'extra_bottleneck_ops/Wx_plus_b/add:0', 'The output tensor name to use.')

tf.app.flags.DEFINE_string(
    'kmeans_tensor', 'InceptionV3/kmeans/Squeeze_2:0', 'The kmeans tensor name to use.')

tf.app.flags.DEFINE_bool(
    'preprocess_image', False, 'Whether to decode and preprocess the image or not.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The batch size the network is supposed to use.')

FLAGS = tf.app.flags.FLAGS

def create_graph(checkpoint):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(checkpoint, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def preprocess_img(filename):
    image = cv2.imread(filename)
    resized = cv2.resize(image, (299, 299)).astype(np.float32)
    resized /= 255
    resized -= 0.5
    resized *= 2
    return resized.reshape((1, 299, 299, 3))

def main(_):
    if not FLAGS.checkpoint:
        raise ValueError('you must supply a checkpoint with --checkpoint')
    if not FLAGS.filelist:
        raise ValueError('you must supply a filelist with --filelist')
    if not FLAGS.save_file:
        raise ValueError('you must supply a save_file with --save_file')

    embedding_list = []
    labels_list = []
    kmeans_list = []
#sess = tf.Session('local', graph=tf.get_default_graph())
#with sess, sess.graph.device(tf.ReplicaDeviceSetter(ps_tasks=0, worker_device='/gpu:0')):
# Creates graph from saved GraphDef.
    create_graph(FLAGS.checkpoint)
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

        #tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        #print(tensor_names)

        embedding_tensor = sess.graph.get_tensor_by_name(FLAGS.output_tensor)
        kmeans_tensor = sess.graph.get_tensor_by_name(FLAGS.kmeans_tensor)

        count = 0
        for line in open(FLAGS.filelist):
            fields = line.split(',')
            if int(fields[3]) == 1:
                labels_list.append(int(fields[2]))

                if FLAGS.preprocess_image:
                    image_data = preprocess_img(fields[0])
                else:
                    image_data = tf.gfile.FastGFile(fields[0], 'rb').read()


                if count == 0 or FLAGS.batch_size == 1:
                    image_stack = image_data
                else:
                    image_stack = np.vstack((image_stack, image_data))

                count += 1
                if count == FLAGS.batch_size or FLAGS.batch_size == 1:
                    count = 0
                    embeddings, k_out = sess.run([embedding_tensor, kmeans_tensor], {FLAGS.input_tensor: image_stack})
                    embedding_list.append(embeddings)
                    kmeans_list.append(k_out)
                c += 1

        # now we need to deal with the case if there are less than batch size images
        if FLAGS.batch_size > 1 and count < FLAGS.batch_size:
            # Pad the image_stack
            num_pad = FLAGS.batch_size - count
            zero_pad = np.zeros((num_pad, 299, 299, 3))
            image_stack = np.vstack((image_stack, zero_pad))
            embeddings, k_out = sess.run([embedding_tensor, kmeans_tensor], {FLAGS.input_tensor: image_stack})
            embedding_list.append(embeddings)
            embedding_list = np.array(embedding_list)
            embedding_list = embedding_list.reshape(c + num_pad, 2048)
            embedding_list = embedding_list[0:c, :]
            kmeans_list.append(k_out)
            kmeans_list = np.array(kmeans_list)
            kmeans_list = kmeans_list.reshape(c + num_pad)
            kmeans_list = kmeans_list[0:c]

    labels_list = np.array(labels_list).reshape(c, 1)
    embedding_list = np.array(embedding_list)
    embedding_list = embedding_list.reshape(c, 2048)
    kmeans_list = np.array(kmeans_list)
    kmeans_list = kmeans_list.reshape(c,1)

    save_array = np.hstack((labels_list, kmeans_list, embedding_list))
    np.save(FLAGS.save_file, save_array)

if __name__ == '__main__':
    tf.app.run()
