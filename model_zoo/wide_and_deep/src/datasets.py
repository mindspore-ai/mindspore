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
"""train_imagenet."""


import os
import math
import numpy as np
import pandas as pd
import mindspore.dataset.engine as de
import mindspore.common.dtype as mstype


class H5Dataset():
    """
    H5DataSet
    """
    input_length = 39

    def __init__(self, data_path, train_mode=True, train_num_of_parts=21,
                 test_num_of_parts=3):
        self._hdf_data_dir = data_path
        self._is_training = train_mode

        if self._is_training:
            self._file_prefix = 'train'
            self._num_of_parts = train_num_of_parts
        else:
            self._file_prefix = 'test'
            self._num_of_parts = test_num_of_parts

        self.data_size = self._bin_count(self._hdf_data_dir, self._file_prefix,
                                         self._num_of_parts)
        print("data_size: {}".format(self.data_size))

    def _bin_count(self, hdf_data_dir, file_prefix, num_of_parts):
        size = 0
        for part in range(num_of_parts):
            _y = pd.read_hdf(os.path.join(hdf_data_dir,
                                          file_prefix + '_output_part_' + str(
                                              part) + '.h5'))
            size += _y.shape[0]
        return size

    def _iterate_hdf_files_(self, num_of_parts=None,
                            shuffle_block=False):
        """
        iterate among hdf files(blocks). when the whole data set is finished, the iterator restarts
            from the beginning, thus the data stream will never stop
        :param train_mode: True or false,false is eval_mode,
            this file iterator will go through the train set
        :param num_of_parts: number of files
        :param shuffle_block: shuffle block files at every round
        :return: input_hdf_file_name, output_hdf_file_name, finish_flag
        """
        parts = np.arange(num_of_parts)
        while True:
            if shuffle_block:
                for _ in range(int(shuffle_block)):
                    np.random.shuffle(parts)
            for i, p in enumerate(parts):
                yield os.path.join(self._hdf_data_dir,
                                   self._file_prefix + '_input_part_' + str(
                                       p) + '.h5'), \
                      os.path.join(self._hdf_data_dir,
                                   self._file_prefix + '_output_part_' + str(
                                       p) + '.h5'), i + 1 == len(parts)

    def _generator(self, X, y, batch_size, shuffle=True):
        """
        should be accessed only in private
        :param X:
        :param y:
        :param batch_size:
        :param shuffle:
        :return:
        """
        number_of_batches = np.ceil(1. * X.shape[0] / batch_size)
        counter = 0
        finished = False
        sample_index = np.arange(X.shape[0])
        if shuffle:
            for _ in range(int(shuffle)):
                np.random.shuffle(sample_index)
        assert X.shape[0] > 0
        while True:
            batch_index = sample_index[
                batch_size * counter: batch_size * (counter + 1)]
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch, finished
            if counter == number_of_batches:
                counter = 0
                finished = True

    def batch_generator(self, batch_size=1000,
                        random_sample=False, shuffle_block=False):
        """
        :param train_mode: True or false,false is eval_mode,
        :param batch_size
        :param num_of_parts: number of files
        :param random_sample: if True, will shuffle
        :param shuffle_block: shuffle file blocks at every round
        :return:
        """

        for hdf_in, hdf_out, _ in self._iterate_hdf_files_(self._num_of_parts,
                                                           shuffle_block):
            start = stop = None
            X_all = pd.read_hdf(hdf_in, start=start, stop=stop).values
            y_all = pd.read_hdf(hdf_out, start=start, stop=stop).values
            data_gen = self._generator(X_all, y_all, batch_size,
                                       shuffle=random_sample)
            finished = False

            while not finished:
                X, y, finished = data_gen.__next__()
                X_id = X[:, 0:self.input_length]
                X_va = X[:, self.input_length:]
                yield np.array(X_id.astype(dtype=np.int32)), np.array(
                    X_va.astype(dtype=np.float32)), np.array(
                        y.astype(dtype=np.float32))


def _get_h5_dataset(data_dir, train_mode=True, epochs=1, batch_size=1000):
    """
    get_h5_dataset
    """
    data_para = {
        'batch_size': batch_size,
    }
    if train_mode:
        data_para['random_sample'] = True
        data_para['shuffle_block'] = True

    h5_dataset = H5Dataset(data_path=data_dir, train_mode=train_mode)
    numbers_of_batch = math.ceil(h5_dataset.data_size / batch_size)

    def _iter_h5_data():
        train_eval_gen = h5_dataset.batch_generator(**data_para)
        for _ in range(0, numbers_of_batch, 1):
            yield train_eval_gen.__next__()

    ds = de.GeneratorDataset(_iter_h5_data(), ["ids", "weights", "labels"])
    ds.set_dataset_size(numbers_of_batch)
    ds = ds.repeat(epochs)
    return ds


def _get_tf_dataset(data_dir, train_mode=True, epochs=1, batch_size=1000,
                    line_per_sample=1000, rank_size=None, rank_id=None):
    """
    get_tf_dataset
    """
    dataset_files = []
    file_prefix_name = 'train' if train_mode else 'test'
    shuffle = train_mode
    for (dirpath, _, filenames) in os.walk(data_dir):
        for filename in filenames:
            if file_prefix_name in filename and "tfrecord" in filename:
                dataset_files.append(os.path.join(dirpath, filename))
    schema = de.Schema()
    schema.add_column('feat_ids', de_type=mstype.int32)
    schema.add_column('feat_vals', de_type=mstype.float32)
    schema.add_column('label', de_type=mstype.float32)
    if rank_size is not None and rank_id is not None:
        ds = de.TFRecordDataset(dataset_files=dataset_files, shuffle=shuffle, schema=schema, num_parallel_workers=8,
                                num_shards=rank_size, shard_id=rank_id, shard_equal_rows=True)
    else:
        ds = de.TFRecordDataset(dataset_files=dataset_files, shuffle=shuffle, schema=schema, num_parallel_workers=8)
    ds = ds.batch(int(batch_size / line_per_sample),
                  drop_remainder=True)
    ds = ds.map(operations=(lambda x, y, z: (
        np.array(x).flatten().reshape(batch_size, 39),
        np.array(y).flatten().reshape(batch_size, 39),
        np.array(z).flatten().reshape(batch_size, 1))),
                input_columns=['feat_ids', 'feat_vals', 'label'],
                columns_order=['feat_ids', 'feat_vals', 'label'], num_parallel_workers=8)
    #if train_mode:
    ds = ds.repeat(epochs)
    return ds


def create_dataset(data_dir, train_mode=True, epochs=1, batch_size=1000,
                   is_tf_dataset=True, line_per_sample=1000, rank_size=None, rank_id=None):
    """
    create_dataset
    """
    if is_tf_dataset:
        return _get_tf_dataset(data_dir, train_mode, epochs, batch_size,
                               line_per_sample, rank_size=rank_size, rank_id=rank_id)
    if rank_size > 1:
        raise RuntimeError("please use tfrecord dataset.")
    return _get_h5_dataset(data_dir, train_mode, epochs, batch_size)
