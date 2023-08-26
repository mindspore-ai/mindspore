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
""" test distribute predict """
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, Model
from mindspore.ops import operations as P
from mindspore import context
from mindspore.parallel._utils import _infer_rank_list


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(128, 768, activation='relu')
        self.fc2 = nn.Dense(128, 768, activation='relu')
        self.fc3 = nn.Dense(128, 768, activation='relu')
        self.fc4 = nn.Dense(768, 768, activation='relu')
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.transpose = P.Transpose()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        k = self.transpose(k, (1, 0))
        c = self.relu4(self.matmul1(q, k))
        s = self.relu5(self.matmul2(c, v))
        s = self.fc4(s)
        return s


def test_distribute_predict():
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True,
                                      enable_parallel_optimizer=True)
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    net = Net()
    model = Model(net)
    predict_map = model.infer_predict_layout(inputs)
    output = model.predict(inputs)
    context.reset_auto_parallel_context()
    return predict_map, output


def test_edge_case():
    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([32, 48]).astype(np.float32))
    net = Net()
    model = Model(net)
    with pytest.raises(RuntimeError):
        model.infer_predict_layout(inputs)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    with pytest.raises(ValueError):
        model.infer_predict_layout(inputs)


# standalone predict
def test_infer_rank_list1():
    train_map = {'weight': [[4, 8], [-1, 0]]}
    predict_map = None
    rank_list = _infer_rank_list(train_map, predict_map)["weight"]
    assert list(rank_list[0]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert rank_list[1] is False


# similar layout: gpt3 prediction mode
def test_infer_rank_list2():
    train_map = {'weight': [[4, 8], [-1, 0]]}
    predict_map = {'weight': [[8], [-1, 0]]}
    rank_list = _infer_rank_list(train_map, predict_map)
    expect_map = {'weight': ([0], True)}
    assert rank_list == expect_map


# same layout
def test_infer_rank_list3():
    train_map = {'weight': [[4, 8], [-1, 0]]}
    predict_map = {'weight': [[4, 8], [-1, 0]]}
    rank_list = _infer_rank_list(train_map, predict_map)
    expect_map = {'weight': ([0], True)}
    assert rank_list == expect_map


# totally different layout
def test_infer_rank_list4():
    train_map = {'weight': [[4, 8], [-1, 0]]}
    predict_map = {'weight': [[2, 2], [1, 0]]}
    rank_list = _infer_rank_list(train_map, predict_map)["weight"]
    assert list(rank_list[0]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert rank_list[1] is False


# full shape ckpt
def test_infer_rank_list5():
    train_map = {'weight': [[8], [-1, -1]]}
    predict_map = {'weight': [[2, 2], [1, 0]]}
    rank_list = _infer_rank_list(train_map, predict_map)
    expect_map = {'weight': ([0], False)}
    assert rank_list == expect_map
