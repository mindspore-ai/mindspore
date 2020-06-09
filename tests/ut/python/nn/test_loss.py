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
""" test loss """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _executor
from ..ut_filter import non_graph_engine


def test_L1Loss():
    loss = nn.L1Loss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
    loss(input_data, target_data)


def test_MSELoss():
    loss = nn.MSELoss()
    input_data = Tensor(np.array([[1, 2, 3], [2, 3, 2]]).astype(np.float32))
    target_data = Tensor(np.array([[0, 0, 5], [1, 2, 3]]).astype(np.float32))
    loss(input_data, target_data)


@non_graph_engine
def test_SoftmaxCrossEntropyWithLogits():
    """ test_SoftmaxCrossEntropyWithLogits """
    loss = nn.SoftmaxCrossEntropyWithLogits()

    logits = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    labels = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    loss.construct(logits, labels)


def test_SoftmaxCrossEntropyWithLogits_reduce():
    """ test_SoftmaxCrossEntropyWithLogits """
    loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")

    logits = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    labels = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    loss(logits, labels)


def test_SoftmaxCrossEntropyExpand():
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE)
    loss = nn.SoftmaxCrossEntropyExpand()

    logits = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    labels = Tensor(np.random.randint(0, 9, [10,]).astype(np.float32))
    _executor.compile(loss, logits, labels)

def test_cosine_embedding_loss():
    """ test CosineEmbeddingLoss """
    loss = nn.CosineEmbeddingLoss()
    x1 = Tensor(np.array([[0.3, 0.8], [0.4, 0.3]]).astype(np.float32))
    x2 = Tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]).astype(np.float32))
    label = Tensor(np.array([1, -1]).astype(np.int32))
    loss(x1, x2, label)
