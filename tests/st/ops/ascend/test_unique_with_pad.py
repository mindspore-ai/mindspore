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
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, pad_num):
        super(Net, self).__init__()
        self.unique_with_pad = P.UniqueWithPad()
        self.pad_num = pad_num

    def construct(self, x):
        return self.unique_with_pad(x, self.pad_num)


def test_unique_with_pad():
    x = Tensor(np.array([1, 1, 5, 5, 4, 4, 3, 3, 2, 2]), mstype.int32)
    pad_num = 8
    unique_with_pad = Net(pad_num)
    out = unique_with_pad(x)
    expect_val = ([1, 5, 4, 3, 2, 8, 8, 8, 8, 8], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert(out[0].asnumpy() == expect_val[0]).all()
    assert(out[1].asnumpy() == expect_val[1]).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unique_with_pad_dynamic_shape():
    """
    Feature: uniquewithpad dynamic shape test in ascend.
    Description: test the rightness of uniquewithpad dynamic shape feature.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 5, 2]).astype(np.int32))
    net = Net(0)
    input_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    net.set_inputs(input_dyn)
    output = net(x)
    expect_y_result = [1, 2, 5, 0]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


def test_unique_with_pad_vmap():
    """
    Feature: uniquewithpad vmap test in ascend.
    Description: test the rightness of uniquewithpad vmap feature.
    Expectation: use vmap rule's result equal to manually batched.
    """

    def cal_unique_with_pad(x):
        return P.UniqueWithPad()(x, -1)

    x = Tensor(np.array([[[1, 2, 5, 2], [1, 2, 5, 2]], [[1, 2, 5, 2], [1, 2, 5, 2]]]).astype(np.int32))

    vmap_unique_with_pad = vmap(vmap(cal_unique_with_pad, in_axes=0), in_axes=0)
    outputs = vmap_unique_with_pad(x)
    expect0 = np.array([[[1, 2, 5, -1], [1, 2, 5, -1]], [[1, 2, 5, -1], [1, 2, 5, -1]]]).astype(np.int32)
    expect1 = np.array([[[0, 1, 2, 1], [0, 1, 2, 1]], [[0, 1, 2, 1], [0, 1, 2, 1]]]).astype(np.int32)
    assert np.allclose(outputs[0].asnumpy(), expect0)
    assert np.allclose(outputs[1].asnumpy(), expect1)
