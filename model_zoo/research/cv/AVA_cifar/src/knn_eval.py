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
"""knn evaluation"""

import math
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P


class KnnEval(nn.Metric):
    """
    collect features for eval
    """

    def __init__(self, batch_size, device_num, K=200, sigma=0.1, C=10, feature_dim=128, train_data_size=50000,
                 test_data_size=10000):
        super(KnnEval, self).__init__()
        self.sum = P.ReduceSum()
        self.batch_size = batch_size
        self.device_num = device_num
        self.feature_dim = feature_dim
        self.train_data_size = train_data_size
        self.test_data_size = test_data_size
        self.K = K
        self.C = C
        self.sigma = sigma
        self.clear()

    def clear(self):
        """clear parameters"""
        self.train_features = np.zeros(
            shape=(self.train_data_size, self.feature_dim), dtype=np.float32)
        self.test_features = np.zeros(
            shape=(self.test_data_size, self.feature_dim), dtype=np.float32)
        self.train_labels = np.zeros(
            shape=(self.train_data_size,), dtype=np.int32)
        self.test_labels = np.zeros(
            shape=(self.test_data_size,), dtype=np.int32)
        self._total_num = 0
        self._total_num_train = 0
        self._total_num_test = 0

    def update(self, *inputs):
        """update"""
        feature = inputs[0].asnumpy()
        label = inputs[1].asnumpy()
        training = inputs[2].asnumpy()
        batch_size = label.shape[0]
        if training.sum() == batch_size:
            self.train_features[self._total_num_train:self._total_num_train +
                                batch_size * self.device_num] = feature
            self.train_labels[self._total_num_train:self._total_num_train +
                              batch_size * self.device_num] = label
            self._total_num_train += batch_size * self.device_num

        elif training.sum() == 0:
            self.test_features[self._total_num_test:self._total_num_test +
                               batch_size * self.device_num] = feature
            self.test_labels[self._total_num_test:self._total_num_test +
                             batch_size * self.device_num] = label
            self._total_num_test += batch_size * self.device_num
        else:
            for i, flag in enumerate(training):
                if flag == 1:
                    self.train_features[self._total_num_train] = feature[i]
                    self.train_labels[self._total_num_train] = label[i]
                    self._total_num_train += 1
                elif flag == 0:
                    self.test_features[self._total_num_test] = feature[i]
                    self.test_labels[self._total_num_test] = label[i]
                    self._total_num_test += 1
        self._total_num = self._total_num_train + self._total_num_test

    def topk(self, matrix, K, axis=1):
        """

        numpy version of torch.topk

        """
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis], dtype=np.int32)
            topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
            topk_data = matrix[topk_index, row_index]
            topk_index_sort = np.argsort(-topk_data, axis=axis)
            topk_data_sort = topk_data[topk_index_sort, row_index]
            topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
        else:
            column_index = np.arange(
                matrix.shape[1 - axis], dtype=np.int32)[:, None]
            topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
            topk_data = matrix[column_index, topk_index]
            topk_index_sort = np.argsort(-topk_data, axis=axis)
            topk_data_sort = topk_data[column_index, topk_index_sort]
            topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
        return topk_data_sort, topk_index_sort

    def gather(self, a, dim, index):
        expanded_index = [
            index if dim == i else np.arange(a.shape[i]).reshape([-1 if i == j else 1 for j in range(a.ndim)]) for i in
            range(a.ndim)]
        return a[expanded_index]

    def scatter(self, a, dim, index, b):
        expanded_index = [
            index if dim == i else np.arange(a.shape[i]).reshape([-1 if i == j else 1 for j in range(a.ndim)]) for i in
            range(a.ndim)]
        a[expanded_index] = b

    def eval(self):
        """compute acc"""
        top1 = 0
        top5 = 0
        for batch_idx in range(math.ceil(len(self.test_labels) / self.batch_size)):
            if batch_idx * self.batch_size > len(self.test_labels):
                test_features = self.test_features[batch_idx *
                                                   self.batch_size:]
                test_labels = self.test_labels[batch_idx * self.batch_size:]
            else:
                test_features = self.test_features[batch_idx *
                                                   self.batch_size:batch_idx * self.batch_size + self.batch_size]
                test_labels = self.test_labels[batch_idx *
                                               self.batch_size:batch_idx * self.batch_size + self.batch_size]

            dist = np.dot(test_features, self.train_features.T)

            yd, yi = self.topk(dist, K=self.K, axis=1)
            candidates = self.train_labels.reshape(
                1, -1).repeat(len(test_labels), 0)  # correct
            retrieval = self.gather(candidates, dim=1, index=yi)
            retrieval = retrieval.astype(np.int32)
            retrieval_one_hot = np.zeros([len(test_labels) * self.K, self.C])
            self.scatter(retrieval_one_hot, 1, retrieval.reshape(-1, 1), 1)

            yd_transform = np.exp(yd / self.sigma)
            retrieval_one_hot = retrieval_one_hot.reshape(
                [len(test_labels), -1, self.C])
            yd_transform = yd_transform.reshape(len(test_labels), -1, 1)
            probs = np.sum(retrieval_one_hot * yd_transform, 1)
            predictions = np.argsort(-probs, 1)
            correct = predictions == test_labels.reshape(-1, 1)
            top1 += np.sum(correct[:, 0:1])
            top5 += np.sum(correct[:, 0:5])
        top1 = top1 / len(self.test_labels)
        top5 = top5 / len(self.test_labels)
        print("top1 acc:{}, top5 acc:{}".format(top1, top5))
        return top1


class FeatureCollectCell(nn.Cell):
    """
    get features from net
    """

    def __init__(self, network):
        super(FeatureCollectCell, self).__init__(auto_prefix=False)
        self._network = network
        self.shape = P.Shape()
        self.sum = P.ReduceSum()

    def construct(self, data, label, training):
        output = self._network(data, data, data)  # redundant input
        feature = output[0]

        return feature, label, training
