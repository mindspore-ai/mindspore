# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Produce the dataset
"""

import mindspore
from mindspore import Tensor
import h5py
import scipy.io as sio
import numpy as np

def dataset_AwA(data_path):
    """input:*.mat, output:array"""
    f = sio.loadmat(data_path+'/AwA_data/train_googlenet_bn.mat')
    train_x_0 = np.array(f['train_googlenet_bn'])

    f = h5py.File(data_path+'/AwA_data/attribute/Z_s_con.mat', 'r')
    train_att_0 = np.array(f['Z_s_con'])

    f = sio.loadmat(data_path+'/AwA_data/wordvector/train_word.mat')
    train_word_0 = np.array(f['train_word'])

    f = sio.loadmat(data_path+'/AwA_data/test_googlenet_bn.mat')
    test_x_0 = np.array(f['test_googlenet_bn'])

    f = sio.loadmat(data_path+'/AwA_data/attribute/pca_te_con_10x85.mat')
    test_att_0 = np.array(f['pca_te_con_10x85'])
    test_att_0 = test_att_0.astype("float16")
    test_att_0 = Tensor(test_att_0, mindspore.float32)

    f = sio.loadmat(data_path+'/AwA_data/wordvector/test_vectors.mat')
    test_word_0 = np.array(f['test_vectors'])
    test_word_0 = test_word_0.astype("float16")
    test_word_0 = Tensor(test_word_0, mindspore.float32)

    f = sio.loadmat(data_path+'/AwA_data/test_labels.mat')
    test_label_0 = np.squeeze(np.array(f['test_labels']))

    f = sio.loadmat(data_path+'/AwA_data/testclasses_id.mat')
    test_id_0 = np.squeeze(np.array(f['testclasses_id']))

    return train_x_0, train_att_0, train_word_0, test_x_0, \
        test_att_0, test_word_0, test_label_0, test_id_0

def dataset_CUB(data_path):
    """input:*.mat, output:array"""
    f = sio.loadmat(data_path+'/CUB_data/train_attr.mat')
    train_att_0 = np.array(f['train_attr'])
    # print('train attr:', train_att.shape)

    f = sio.loadmat(data_path+'/CUB_data/train_cub_googlenet_bn.mat')
    train_x_0 = np.array(f['train_cub_googlenet_bn'])
    # print('train x:', train_x.shape)

    f = sio.loadmat(data_path+'/CUB_data/test_cub_googlenet_bn.mat')
    test_x_0 = np.array(f['test_cub_googlenet_bn'])
    # print('test x:', test_x.shape)

    f = sio.loadmat(data_path+'/CUB_data/test_proto.mat')
    test_att_0 = np.array(f['test_proto'])
    test_att_0 = test_att_0.astype("float16")
    test_att_0 = Tensor(test_att_0, mindspore.float32)
    # print('test att:', test_att.shape)

    f = sio.loadmat(data_path+'/CUB_data/test_labels_cub.mat')
    test_label_0 = np.squeeze(np.array(f['test_labels_cub']))
    # print('test x2label:', test_x2label)

    f = sio.loadmat(data_path+'/CUB_data/testclasses_id.mat')
    test_id_0 = np.squeeze(np.array(f['testclasses_id']))
    # print('test att2label:', test_att2label)

    return train_att_0, train_x_0, test_x_0, test_att_0, test_label_0, test_id_0

class SingleDataIterable:
    """data+label"""
    def __init__(self, data, label):
        self._data = data
        self._label = label

    def __getitem__(self, index):
        item1 = self._data[index:index + 1]
        item2 = self._label[index:index + 1]
        return item1.astype(np.float32), item2.astype(np.float32)

    def __len__(self):
        return len(self._data)

class DoubleDataIterable:
    """data1+data2+label"""
    def __init__(self, data1, data2, label):
        self._data1 = data1
        self._data2 = data2
        self._label = label

    def __getitem__(self, index):
        item1 = self._data1[index:index + 1]
        item2 = self._data2[index:index + 1]
        item3 = self._label[index:index + 1]
        return item1.astype(np.float32), item2.astype(np.float32), item3.astype(np.float32)

    def __len__(self):
        return len(self._data1)

if __name__ == "__main__":
    train_att, train_x, test_x, test_att, test_label, test_id = dataset_CUB('/data/DEM_data')
    print('train attr:', train_att.shape)
    print('train x:', train_x.shape)
    print('test x:', test_x.shape)
    print('test att:', test_att.shape)
    print('test label:', test_label)
    print('test id:', test_id)
