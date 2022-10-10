# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.ops as op
from mindspore.nn import Cell
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import functional as F


class BatchMatMulNet(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(BatchMatMulNet, self).__init__()
        self.batch_matmul = P.BatchMatMul(transpose_a, transpose_b)

    def construct(self, x, y):
        return self.batch_matmul(x, y)


class BatchMatMulDynamicRank(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(BatchMatMulDynamicRank, self).__init__()
        self.op = P.BatchMatMul(transpose_a, transpose_b)
        self.reduce_sum = P.ReduceSum(keep_dims=False)

    def construct(self, x, y, dyn_reduce_axis):
        x = self.reduce_sum(x, dyn_reduce_axis)
        y = self.reduce_sum(y, dyn_reduce_axis)
        res = self.op(x, y)
        return res


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4d():
    x = np.arange(2 * 4 * 1 * 3).reshape((2, 4, 1, 3)).astype(np.float32)
    y = np.arange(2 * 4 * 3 * 4).reshape((2, 4, 3, 4)).astype(np.float32)
    input_x = Tensor(x, mstype.float32)
    input_y = Tensor(y, mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet()
    output = net(input_x, input_y)
    expect = [[[[20, 23, 26, 29]],
               [[200, 212, 224, 236]],
               [[596, 617, 638, 659]],
               [[1208, 1238, 1268, 1298]]],

              [[[2036, 2075, 2114, 2153]],
               [[3080, 3128, 3176, 3224]],
               [[4340, 4397, 4454, 4511]],
               [[5816, 5882, 5948, 6014]]]]
    assert (output.asnumpy() == expect).all()

    # test dynamic_shape
    dyn_shape_net = BatchMatMulNet()
    input_x_dyn = Tensor(shape=[2, None, 1, 3], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, None, 3, 4], dtype=mstype.float32)
    dyn_shape_net.set_inputs(input_x_dyn, input_y_dyn)
    output = dyn_shape_net(input_x, input_y)
    assert (output.asnumpy() == expect).all()

    # test dynamic_rank
    dyn_rank_net = BatchMatMulDynamicRank()
    input_x_dyn = Tensor(shape=[2, None, 1, 3, 1], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, None, 3, 4, 1], dtype=mstype.float32)
    dyn_reduce_axis = Tensor(shape=[None], dtype=mstype.int64)
    dyn_rank_net.set_inputs(input_x_dyn, input_y_dyn, dyn_reduce_axis)

    reduce_axis = np.array([-1], dtype=np.int64)
    output = dyn_rank_net(Tensor(np.expand_dims(x, -1)),
                          Tensor(np.expand_dims(y, -1)), Tensor(reduce_axis))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4d_float64():
    input_x = Tensor(np.arange(2 * 4 * 1 * 3).reshape(2, 4, 1, 3), mstype.float64)
    input_y = Tensor(np.arange(2 * 4 * 3 * 4).reshape(2, 4, 3, 4), mstype.float64)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet()
    output = net(input_x, input_y)
    expect = [[[[20, 23, 26, 29]],
               [[200, 212, 224, 236]],
               [[596, 617, 638, 659]],
               [[1208, 1238, 1268, 1298]]],

              [[[2036, 2075, 2114, 2153]],
               [[3080, 3128, 3176, 3224]],
               [[4340, 4397, 4454, 4511]],
               [[5816, 5882, 5948, 6014]]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4d_transpose_a():
    input_x = Tensor(np.arange(2 * 4 * 3 * 1).reshape(2, 4, 3, 1), mstype.float32)
    input_y = Tensor(np.arange(2 * 4 * 3 * 4).reshape(2, 4, 3, 4), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet(transpose_a=True)
    output = net(input_x, input_y)
    expect = [[[[20, 23, 26, 29]],
               [[200, 212, 224, 236]],
               [[596, 617, 638, 659]],
               [[1208, 1238, 1268, 1298]]],

              [[[2036, 2075, 2114, 2153]],
               [[3080, 3128, 3176, 3224]],
               [[4340, 4397, 4454, 4511]],
               [[5816, 5882, 5948, 6014]]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4d_transpose_b():
    input_x = Tensor(np.arange(2 * 4 * 1 * 3).reshape(2, 4, 1, 3), mstype.float32)
    input_y = Tensor(np.arange(2 * 4 * 4 * 3).reshape(2, 4, 4, 3), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet(transpose_b=True)
    output = net(input_x, input_y)
    expect = [[[[5, 14, 23, 32]],
               [[158, 194, 230, 266]],
               [[527, 590, 653, 716]],
               [[1112, 1202, 1292, 1382]]],

              [[[1913, 2030, 2147, 2264]],
               [[2930, 3074, 3218, 3362]],
               [[4163, 4334, 4505, 4676]],
               [[5612, 5810, 6008, 6206]]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4d_transpose_ab():
    input_x = Tensor(np.arange(2 * 4 * 3 * 1).reshape(2, 4, 3, 1), mstype.float32)
    input_y = Tensor(np.arange(2 * 4 * 4 * 3).reshape(2, 4, 4, 3), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet(transpose_a=True, transpose_b=True)
    output = net(input_x, input_y)
    expect = [[[[5, 14, 23, 32]],
               [[158, 194, 230, 266]],
               [[527, 590, 653, 716]],
               [[1112, 1202, 1292, 1382]]],

              [[[1913, 2030, 2147, 2264]],
               [[2930, 3074, 3218, 3362]],
               [[4163, 4334, 4505, 4676]],
               [[5612, 5810, 6008, 6206]]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4d_fp_16():
    """
    Feature: test BatchMatMul op.
    Description: test BatchMatMul 4d input dtype float16.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.arange(2 * 4 * 1 * 3).reshape(2, 4, 1, 3), mstype.float16)
    input_y = Tensor(np.arange(2 * 4 * 3 * 4).reshape(2, 4, 3, 4), mstype.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet()
    output = net(input_x, input_y)
    expect = np.array([[[[20, 23, 26, 29]],
                        [[200, 212, 224, 236]],
                        [[596, 617, 638, 659]],
                        [[1208, 1238, 1268, 1298]]],

                       [[[2036, 2076, 2114, 2152]],
                        [[3080, 3128, 3176, 3224]],
                        [[4340, 4396, 4456, 4510]],
                        [[5816, 5880, 5948, 6016]]]]).astype(np.float16)
    assert (output.asnumpy() == expect).all()


class BatchMatMulDynamic(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(BatchMatMulDynamic, self).__init__()
        self.batch_matmul = P.BatchMatMul(transpose_a, transpose_b)
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, x, y):
        x = self.test_dynamic(x)
        y = self.test_dynamic(y)
        return self.batch_matmul(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_dynamic():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulDynamic()

    x1 = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    y1 = np.arange(28).reshape(2, 2, 7).astype(np.float32)

    output1 = net(Tensor(x1), Tensor(y1))
    expect1 = np.matmul(x1, y1)
    assert (output1.asnumpy() == expect1).all()

    x2 = np.arange(2 * 4 * 1 * 3).reshape(2, 4, 1, 3).astype(np.float32)
    y2 = np.arange(2 * 4 * 3 * 4).reshape(2, 4, 3, 4).astype(np.float32)

    output2 = net(Tensor(x2), Tensor(y2))
    expect2 = np.matmul(x2, y2)
    assert (output2.asnumpy() == expect2).all()


def test_bmm_forward_tensor_api(nptype):
    """
    Feature: test bmm forward tensor api for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones(shape=[2, 4, 1, 3]).astype(nptype))
    y = Tensor(np.ones(shape=[2, 4, 3, 4]).astype(nptype))
    output = x.bmm(y)
    expected = 3 * np.ones(shape=[2, 4, 1, 4]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bmm_forward_float32_tensor_api():
    """
    Feature: test bmm forward tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_bmm_forward_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_bmm_forward_tensor_api(np.float32)


def test_bmm_forward_functional_api(nptype):
    """
    Feature: test bmm forward functional api for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones(shape=[2, 4, 1, 3]).astype(nptype))
    y = Tensor(np.ones(shape=[2, 4, 3, 4]).astype(nptype))
    output = F.bmm(x, y)
    expected = 3 * np.ones(shape=[2, 4, 1, 4]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bmm_forward_float32_functional_api():
    """
    Feature: test bmm forward functional api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_bmm_forward_functional_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_bmm_forward_functional_api(np.float32)


class BatchMatMul(Cell):
    def __init__(self):
        super().__init__()
        self.batchmatmul = op.matmul

    def construct(self, x1, x2):
        return self.batchmatmul(x1, x2)


class BatchMatMulTestNet(Cell):
    def __init__(self, inputs=None):
        self.input_x1 = inputs[0]
        self.input_x2 = inputs[1]

    def forward_mindspore_impl(self):
        input_x1 = Tensor(self.input_x1)
        input_x2 = Tensor(self.input_x2)
        net = BatchMatMul()
        out = net(input_x1, input_x2)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_type_float16():
    """
    Feature: test bmm with dtype float16.
    Description: test bmm with dtype float16.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.float16)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.float16)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    float16_out = fact.forward_mindspore_impl()
    expect = np.array([[[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]]], np.float16)
    assert (float16_out.asnumpy() == expect).all()
    assert str(float16_out.dtype) == "Float16"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_type_float32():
    """
    Feature: test bmm with dtype float32.
    Description: test bmm with dtype float32.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.float32)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.float32)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    float32_out = fact.forward_mindspore_impl()
    expect = np.array([[[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]]], np.float32)
    assert (float32_out.asnumpy() == expect).all()
    assert str(float32_out.dtype) == "Float32"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_type_float64():
    """
    Feature: test bmm with dtype float64.
    Description: test bmm with dtype float64.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.float64)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.float64)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    float64_out = fact.forward_mindspore_impl()
    expect = np.array([[[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]],
                       [[3., 3., 3., 3.]]], np.float64)
    assert (float64_out.asnumpy() == expect).all()
    assert str(float64_out.dtype) == "Float64"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_type_int8():
    """
    Feature: test bmm with dtype int8.
    Description: test bmm with dtype int8.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 4]), ms.int8)
    input_x2 = Tensor(np.ones(shape=[4, 4, 4]), ms.int8)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    out = fact.forward_mindspore_impl()
    expect = np.array([[[4, 4, 4, 4]],
                       [[4, 4, 4, 4]],
                       [[4, 4, 4, 4]],
                       [[4, 4, 4, 4]]], np.int32)
    assert (out.asnumpy() == expect).all()
    assert str(out.dtype) == "Int32"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_type_complex64():
    """
    Feature: test bmm with dtype complex64.
    Description: test bmm with dtype complex64.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_list = []
    input_x1 = Tensor(np.array([[[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]],
                                [[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]],
                                [[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]],
                                [[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]]]).astype(np.complex64))
    input_x2 = Tensor(np.array([[[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]],
                                [[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]],
                                [[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]],
                                [[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]]]).astype(np.complex64))
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    out = fact.forward_mindspore_impl()
    expect = np.array([[[8+0j, 8+0j, 8+0j, 8+0j]],
                       [[8+0j, 8+0j, 8+0j, 8+0j]],
                       [[8+0j, 8+0j, 8+0j, 8+0j]],
                       [[8+0j, 8+0j, 8+0j, 8+0j]]], np.complex64)
    assert (out.asnumpy() == expect).all()
    assert str(out.dtype) == "Complex64"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_type_complex128():
    """
    Feature: test bmm with dtype complex128.
    Description: test bmm with dtype complex128.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_list = []
    input_x1 = Tensor(np.array([[[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]],
                                [[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]],
                                [[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]],
                                [[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]]]).astype(np.complex128))
    input_x2 = Tensor(np.array([[[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]],
                                [[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]],
                                [[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]],
                                [[1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j],
                                 [1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j]]]).astype(np.complex128))
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    out = fact.forward_mindspore_impl()
    expect = np.array([[[8+0j, 8+0j, 8+0j, 8+0j]],
                       [[8+0j, 8+0j, 8+0j, 8+0j]],
                       [[8+0j, 8+0j, 8+0j, 8+0j]],
                       [[8+0j, 8+0j, 8+0j, 8+0j]]], np.complex128)
    assert (out.asnumpy() == expect).all()
    assert str(out.dtype) == "Complex128"
