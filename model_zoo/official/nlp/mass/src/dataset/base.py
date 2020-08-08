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
"""Base class of data loader."""
import os
import collections
import numpy as np

from mindspore.mindrecord import FileWriter
from .schema import SCHEMA


class DataLoader:
    """Data loader for dataset."""
    _SCHEMA = SCHEMA

    def __init__(self, max_sen_len=66):
        self._examples = []
        self._max_sentence_len = max_sen_len

    def _load(self):
        raise NotImplementedError

    def padding(self, sen, padding_idx, dtype=np.int64):
        """Padding <pad> to sentence."""
        if sen.shape[0] > self._max_sentence_len:
            return None
        new_sen = np.array([padding_idx] * self._max_sentence_len,
                           dtype=dtype)
        new_sen[:sen.shape[0]] = sen[:]
        return new_sen

    def write_to_mindrecord(self, path, shard_num=1, desc=""):
        """
        Write mindrecord file.

        Args:
            path (str): File path.
            shard_num (int): Shard num.
            desc (str): Description.
        """
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        writer = FileWriter(file_name=path, shard_num=shard_num)
        writer.add_schema(self._SCHEMA, desc)
        if not self._examples:
            self._load()

        writer.write_raw_data(self._examples)
        writer.commit()
        print(f"| Wrote to {path}.")

    def write_to_tfrecord(self, path, shard_num=1):
        """
        Write to tfrecord.

        Args:
            path (str): Output file path.
            shard_num (int): Shard num.
        """
        import tensorflow as tf
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        output_files = []
        for i in range(shard_num):
            output_file = path + "-%03d-of-%03d" % (i + 1, shard_num)
            output_files.append(output_file)
        # create writers
        writers = []
        for output_file in output_files:
            writers.append(tf.io.TFRecordWriter(output_file))

        if not self._examples:
            self._load()

        # create feature
        features = collections.OrderedDict()
        for example in self._examples:
            for key in example:
                features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=example[key].tolist()))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            for writer in writers:
                writer.write(tf_example.SerializeToString())
        for writer in writers:
            writer.close()
        for p in output_files:
            print(f" | Write to {p}.")

    def _add_example(self, example):
        self._examples.append(example)
