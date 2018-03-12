# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/train/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/eval/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cityscapes', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3_kmeans', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_boolean(
    'allow_gpu_growth', False,
    'Whether to set the allow_growth flag to minimize gpu memory use.')

tf.app.flags.DEFINE_integer('kmeans', 0,
                     'The number of k for kmeans (0 means dont use).')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label, orig_label] = provider.get(['image', 'label', 'orig_label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels, orig_labels = tf.train.batch(
        [image, label, orig_label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images, kmeans_num_k=FLAGS.kmeans, binary_labels=labels)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    })

    with tf.variable_scope('Accuracy'):
      labels_when = tf.equal(labels, tf.constant(0, dtype=tf.int64))
      ind = tf.where(labels_when)
      new_pred = tf.gather(predictions, ind)
      new_acc = tf.contrib.metrics.accuracy(tf.zeros_like(new_pred), new_pred)
      tf.summary.scalar('acc_for_label_0', new_acc)
      foreground_labels = [11, 21, 24, 25, 26, 27, 28, 31, 32, 33]
      for f in foreground_labels:
        labels_when = tf.equal(orig_labels, tf.constant(f, dtype=tf.int64))
        ind = tf.where(labels_when)
        new_pred = tf.gather(predictions, ind)
        new_acc = tf.contrib.metrics.accuracy(new_pred, tf.ones_like(new_pred))
        tf.summary.scalar('acc_for_label_' + str(f), new_acc)
        # Let's put our Kmeans metrics here:
        with tf.variable_scope('KmeansAccuracy'):
          for i in range(0, FLAGS.kmeans):
            kmeans_pred = tf.gather(end_points['KClusters'], ind)
            if i == 0:
              kmeans_acc = tf.contrib.metrics.accuracy(
                  kmeans_pred, tf.zeros_like(kmeans_pred))
            else:
              kmeans_acc = tf.contrib.metrics.accuracy(
                  kmeans_pred,
                  tf.scalar_mul(tf.constant(i), tf.ones_like(kmeans_pred)))
            tf.summary.scalar('k_' + str(i) + '_for_class_' + str(f),
                              kmeans_acc)

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    #if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    #  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    #else:
    #  checkpoint_path = FLAGS.checkpoint_path
    checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Evaluating %s' % checkpoint_path)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # Check whether to minimize GPU memory use with allow_growth
    if FLAGS.allow_gpu_growth:
      sess_config.gpu_options.allow_growth = True

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore,
        session_config=sess_config)


if __name__ == '__main__':
  tf.app.run()
