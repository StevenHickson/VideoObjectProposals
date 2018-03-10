# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Cityscapes data to TFRecords of TF-Example protos.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_string(
    'train_file',
    None,
    'The train file.')

tf.app.flags.DEFINE_string(
    'val_file',
    None,
    'The val file.')

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'cityscapes_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filename, dataset_dir):
  """Converts the given filelist to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filelist: A list of absolute paths to png, the label, and the original label.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  filelist = []
  for line in open(filename):
    filelist.append(line)
  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(filelist)


  num_per_shard = int(math.ceil(len(filelist) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filelist))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filelist), shard_id))
            sys.stdout.flush()

            # Let's parse the filelist
            fields = filelist[i].split(',')
            filename = fields[0]
            orig_label = int(fields[2])
            label = int(fields[3])

            # Read the filename:
            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            example = dataset_utils.image_and_labels_to_tfexample(
                image_data, b'png', height, width, label, orig_label)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def run(dataset_dir, train_file, val_file):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)


  # First, convert the training and validation sets.
  _convert_dataset('train', train_file, dataset_dir)
  _convert_dataset('validation', val_file, dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = {0: 'background', 1: 'foreground'}
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the cityscapes dataset!')


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  
  run(FLAGS.dataset_dir, FLAGS.train_file, FLAGS.val_file)

if __name__ == '__main__':
  tf.app.run()
