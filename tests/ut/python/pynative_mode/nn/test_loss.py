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
""" test_loss """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from ...ut_filter import non_graph_engine


def test_L1Loss():
    loss = nn.L1Loss()
    input_data = Tensor(np.array([1, 2, 3]))
    target_data = Tensor(np.array([1, 2, 2]))
    with pytest.raises(NotImplementedError):
        loss.construct(input_data, target_data)


@non_graph_engine
def test_SoftmaxCrossEntropyWithLogits():
    """ test_SoftmaxCrossEntropyWithLogits """
    loss = nn.SoftmaxCrossEntropyWithLogits()

    logits = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    labels = Tensor(np.random.randint(0, 9, [100, 10]).astype(np.float32))
    loss.construct(logits, labels)
