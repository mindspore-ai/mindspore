# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
"""test dataset performance about mindspore.MindDataset, mindspore.TFRecordDataset, tf.data.TFRecordDataset"""
import time
import tensorflow as tf

import mindspore.dataset as ds
from mindspore.mindrecord import FileReader

print_step = 5000


def print_log(count):
    if count % print_step == 0:
        print("Read {} rows ...".format(count))


def use_filereader(mindrecord):
    start = time.time()
    columns_list = ["data", "label"]
    reader = FileReader(file_name=mindrecord,
                        num_consumer=4,
                        columns=columns_list)
    num_iter = 0
    for _, _ in enumerate(reader.get_next()):
        num_iter += 1
        print_log(num_iter)
    end = time.time()
    print("Read by FileReader - total rows: {}, cost time: {}s".format(num_iter, end - start))


def use_minddataset(mindrecord):
    start = time.time()
    columns_list = ["data", "label"]
    data_set = ds.MindDataset(dataset_file=mindrecord,
                              columns_list=columns_list,
                              num_parallel_workers=4)
    num_iter = 0
    for _ in data_set.create_dict_iterator(num_epochs=1):
        num_iter += 1
        print_log(num_iter)
    end = time.time()
    print("Read by MindDataset - total rows: {}, cost time: {}s".format(num_iter, end - start))


def use_tfrecorddataset(tfrecord):
    start = time.time()
    columns_list = ["data", "label"]
    data_set = ds.TFRecordDataset(dataset_files=tfrecord,
                                  columns_list=columns_list,
                                  num_parallel_workers=4,
                                  shuffle=ds.Shuffle.GLOBAL)
    data_set = data_set.shuffle(10000)
    num_iter = 0
    for _ in data_set.create_dict_iterator(num_epochs=1):
        num_iter += 1
        print_log(num_iter)
    end = time.time()
    print("Read by TFRecordDataset - total rows: {}, cost time: {}s".format(num_iter, end - start))


def use_tensorflow_tfrecorddataset(tfrecord):
    start = time.time()

    def _parse_record(example_photo):
        features = {
            'file_name': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string)}
        parsed_features = tf.io.parse_single_example(example_photo, features=features)
        return parsed_features

    data_set = tf.data.TFRecordDataset(filenames=tfrecord,
                                       buffer_size=100000,
                                       num_parallel_reads=4)
    data_set = data_set.map(_parse_record, num_parallel_calls=4)
    num_iter = 0
    for _ in data_set.__iter__():
        num_iter += 1
        print_log(num_iter)
    end = time.time()
    print("Read by TensorFlow TFRecordDataset - total rows: {}, cost time: {}s".format(num_iter, end - start))


if __name__ == '__main__':
    # use MindDataset
    mindrecord_test = './imagenet.mindrecord00'
    use_minddataset(mindrecord_test)

    # use TFRecordDataset
    tfrecord_test = ['imagenet.tfrecord00', 'imagenet.tfrecord01', 'imagenet.tfrecord02', 'imagenet.tfrecord03',
                     'imagenet.tfrecord04', 'imagenet.tfrecord05', 'imagenet.tfrecord06', 'imagenet.tfrecord07',
                     'imagenet.tfrecord08', 'imagenet.tfrecord09', 'imagenet.tfrecord10', 'imagenet.tfrecord11',
                     'imagenet.tfrecord12', 'imagenet.tfrecord13', 'imagenet.tfrecord14', 'imagenet.tfrecord15']
    use_tfrecorddataset(tfrecord_test)

    # use TensorFlow TFRecordDataset
    use_tensorflow_tfrecorddataset(tfrecord_test)

    # use FileReader
    # use_filereader(mindrecord)
