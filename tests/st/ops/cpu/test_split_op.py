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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out_float32():
    op = P.Split(5, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(192).astype(np.float32).reshape((2, 2, 2, 2, 2, 6)))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0., 1., 2.])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [3., 4., 5.])

    op = P.Split(5, 3)
    op_wrapper = OpNetWrapper(op)
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0., 1.])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [2., 3.])
    assert np.allclose(outputs[2].asnumpy()[0, 0, 0, 0, 0, :], [4., 5.])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out_float64():
    op = P.Split(5, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(192).astype(np.float64).reshape((2, 2, 2, 2, 2, 6)))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0., 1., 2.])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [3., 4., 5.])

    op = P.Split(5, 3)
    op_wrapper = OpNetWrapper(op)
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0., 1.])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [2., 3.])
    assert np.allclose(outputs[2].asnumpy()[0, 0, 0, 0, 0, :], [4., 5.])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out_float16():
    op = P.Split(-1, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(320).astype(np.float16).reshape((2, 2, 2, 2, 2, 10)))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0., 1., 2., 3., 4.])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [5., 6., 7., 8., 9.])

    op = P.Split(-1, 5)
    op_wrapper = OpNetWrapper(op)
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0., 1.])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [2., 3.])
    assert np.allclose(outputs[2].asnumpy()[0, 0, 0, 0, 0, :], [4., 5.])
    assert np.allclose(outputs[3].asnumpy()[0, 0, 0, 0, 0, :], [6., 7.])
    assert np.allclose(outputs[4].asnumpy()[0, 0, 0, 0, 0, :], [8., 9.])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out_int32():
    op = P.Split(5, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(192).astype(np.int32).reshape((2, 2, 2, 2, 2, 6)))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0, 1, 2])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [3, 4, 5])

    op = P.Split(5, 3)
    op_wrapper = OpNetWrapper(op)
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[1, 0, 0, 0, 0, :], [96, 97])
    assert np.allclose(outputs[1].asnumpy()[1, 0, 0, 0, 0, :], [98, 99])
    assert np.allclose(outputs[2].asnumpy()[1, 0, 0, 0, 0, :], [100, 101])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out_int64():
    op = P.Split(5, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(192).astype(np.int64).reshape((2, 2, 2, 2, 2, 6)))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0, 1, 2])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [3, 4, 5])

    op = P.Split(5, 3)
    op_wrapper = OpNetWrapper(op)
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[1, 0, 0, 0, 0, :], [96, 97])
    assert np.allclose(outputs[1].asnumpy()[1, 0, 0, 0, 0, :], [98, 99])
    assert np.allclose(outputs[2].asnumpy()[1, 0, 0, 0, 0, :], [100, 101])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_out_uint32():
    op = P.Split(-1, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(320).astype(np.uint32).reshape((2, 2, 2, 2, 2, 10)))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, 0, :], [0, 1, 2, 3, 4])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, 0, :], [5, 6, 7, 8, 9])

    op = P.Split(-1, 5)
    op_wrapper = OpNetWrapper(op)
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[1, 1, 1, 1, 1, :], [310, 311])
    assert np.allclose(outputs[1].asnumpy()[1, 1, 1, 1, 1, :], [312, 313])
    assert np.allclose(outputs[2].asnumpy()[1, 1, 1, 1, 1, :], [314, 315])
    assert np.allclose(outputs[3].asnumpy()[1, 1, 1, 1, 1, :], [316, 317])
    assert np.allclose(outputs[4].asnumpy()[1, 1, 1, 1, 1, :], [318, 319])

    op = P.Split(-2, 2)
    op_wrapper = OpNetWrapper(op)
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy()[0, 0, 0, 0, :, 0], [0])
    assert np.allclose(outputs[1].asnumpy()[0, 0, 0, 0, :, 1], [11])
    assert np.allclose(outputs[0].asnumpy()[1, 0, 0, 0, :, 2], [162])
    assert np.allclose(outputs[1].asnumpy()[1, 0, 0, 0, :, 3], [173])
    assert np.allclose(outputs[0].asnumpy()[1, 1, 0, 0, :, 4], [244])
    assert np.allclose(outputs[1].asnumpy()[1, 1, 0, 0, :, 5], [255])
    assert np.allclose(outputs[0].asnumpy()[1, 1, 1, 0, :, 6], [286])
    assert np.allclose(outputs[1].asnumpy()[1, 1, 1, 0, :, 7], [297])
    assert np.allclose(outputs[0].asnumpy()[1, 1, 1, 1, :, 8], [308])
    assert np.allclose(outputs[1].asnumpy()[1, 1, 1, 1, :, 9], [319])

    op = P.Split(-1, 1)
    op_wrapper = OpNetWrapper(op)
    input_x = Tensor(np.arange(1).astype(np.uint32))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs[0].asnumpy(), [0])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_split_dynamic_shape():
    """
    Feature: Split ops with dynamic shape
    Description: test cases with dynamic shape inputs
    Expectation: success
    """
    op = P.Split(-1, 2)
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.arange(192).astype(np.int32).reshape((2, 2, 2, 2, 2, 6)))
    outputs = op_wrapper(input_x)

    dyn_input = Tensor(shape=[2, 2, 2, 2, 2, None], dtype=mstype.int32)
    op_wrapper.set_inputs(dyn_input)
    dyn_outputs = op_wrapper(input_x)

    assert (outputs[0].asnumpy() == dyn_outputs[0].asnumpy()).all()
    assert (outputs[1].asnumpy() == dyn_outputs[1].asnumpy()).all()
