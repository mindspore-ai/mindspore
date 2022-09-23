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
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = P.EmbeddingLookup().set_device("CPU")

    def construct(self, param, index, offset):
        return self.embedding(param, index, offset)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_embedding_look_up0():
    params = Tensor(
        np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mstype.float32)
    indices = Tensor(np.array([5, 2, 8, 5]), mstype.int32)
    offset = 4
    embedding = Net()
    out = embedding(params, indices, offset)
    print(out)
    expect = np.array([[10, 11], [0, 0], [0, 0], [10, 11]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_embedding_look_up1():
    params = Tensor(np.array([[8, 9], [10, 11]]), mstype.float32)
    indices = Tensor(np.array([2, 2, 1, 0]), mstype.int32)
    offset = 0
    embedding = Net()
    out = embedding(params, indices, offset)
    expect = np.array([[0, 0], [0, 0], [10, 11], [8, 9]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_embedding_look_up2():
    params = Tensor(
        np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mstype.float32)
    indices = Tensor(np.array([[5, 2], [8, 5]]), mstype.int32)
    offset = 4
    embedding = Net()
    out = embedding(params, indices, offset)
    expect = np.array(
        [[[10, 11], [0, 0]], [[0, 0], [10, 11]]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_embedding_look_up3():
    params = Tensor(
        np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mstype.float32)
    indices = Tensor(np.array([[[5], [2]], [[8], [5]]]), mstype.int32)
    offset = 4
    embedding = Net()
    out = embedding(params, indices, offset)
    expect = np.array(
        [[[[10, 11]], [[0, 0]]], [[[0, 0]], [[10, 11]]]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()
