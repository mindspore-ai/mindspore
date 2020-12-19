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
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class OpNetWrapper(nn.Cell):
    def __init__(self, op):
        super(OpNetWrapper, self).__init__()
        self.op = op

    def construct(self, *inputs):
        return self.op(*inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out1_axis0():
    op = P.Split(0, 1)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(24).astype(np.int32).reshape((2, 2, 6)))
    outputs = op_wrapper(input_x)

    print(outputs)
    assert outputs[0].shape == (2, 2, 6)
    assert np.allclose(outputs[0].asnumpy()[0, 0, :], [0, 1, 2, 3, 4, 5])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out2_axis2():
    op = P.Split(2, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(24).astype(np.int32).reshape((2, 2, 6)))
    outputs = op_wrapper(input_x)

    print(outputs)
    assert outputs[0].shape == (2, 2, 3)
    assert outputs[1].shape == (2, 2, 3)
    assert np.allclose(outputs[0].asnumpy()[0, 0, :], [0, 1, 2])
    assert np.allclose(outputs[1].asnumpy()[0, 0, :], [3, 4, 5])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out2_axis1neg():
    op = P.Split(-1, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(24).astype(np.float32).reshape((2, 2, 6)))
    outputs = op_wrapper(input_x)

    print(outputs)
    assert np.allclose(outputs[0].asnumpy()[0, :, :], [[0., 1., 2.], [6., 7., 8.]])
    assert np.allclose(outputs[1].asnumpy()[0, :, :], [[3., 4., 5.], [9., 10., 11.]])


if __name__ == '__main__':
    test_out1_axis0()
    test_out2_axis2()
    test_out2_axis1neg()
