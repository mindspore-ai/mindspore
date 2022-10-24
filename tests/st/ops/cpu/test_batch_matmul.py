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
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as op
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
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


def judge_result_correct(result, expect):
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_4d_no_transpose_vec():
    x = np.arange(2 * 4 * 1 * 3).reshape((2, 4, 1, 3)).astype(np.float32)
    y = np.arange(2 * 4 * 3 * 4).reshape((2, 4, 3, 4)).astype(np.float32)
    input_x = Tensor(x, mstype.float32)
    input_y = Tensor(y, mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = BatchMatMulNet()
    output = net(input_x, input_y)
    expect = np.array([[[[20, 23, 26, 29]],
                        [[200, 212, 224, 236]],
                        [[596, 617, 638, 659]],
                        [[1208, 1238, 1268, 1298]]],
                       [[[2036, 2075, 2114, 2153]],
                        [[3080, 3128, 3176, 3224]],
                        [[4340, 4397, 4454, 4511]],
                        [[5816, 5882, 5948, 6014]]]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)

    # test dynamic_shape
    dyn_shape_net = BatchMatMulNet()
    input_x_dyn = Tensor(shape=[2, None, 1, 3], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, None, 3, 4], dtype=mstype.float32)
    dyn_shape_net.set_inputs(input_x_dyn, input_y_dyn)
    output = dyn_shape_net(input_x, input_y)
    judge_result_correct(output.asnumpy(), expect)

    # test dynamic_rank
    dyn_rank_net = BatchMatMulDynamicRank()
    input_x_dyn = Tensor(shape=[2, None, 1, 3, 1], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, None, 3, 4, 1], dtype=mstype.float32)
    dyn_reduce_axis = Tensor(shape=[None], dtype=mstype.int64)
    dyn_rank_net.set_inputs(input_x_dyn, input_y_dyn, dyn_reduce_axis)

    reduce_axis = np.array([-1], dtype=np.int64)
    output = dyn_rank_net(Tensor(np.expand_dims(x, -1)),
                          Tensor(np.expand_dims(y, -1)), Tensor(reduce_axis))
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_4d_no_transpose():
    input_x = Tensor(np.arange(2 * 3 * 2 * 3).reshape((2, 3, 2, 3)), mstype.float32)
    input_y = Tensor(np.arange(2 * 3 * 3 * 4).reshape((2, 3, 3, 4)), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = BatchMatMulNet()
    output = net(input_x, input_y)
    expect = np.array([[[[20., 23., 26., 29.],
                         [56., 68., 80., 92.]],
                        [[344., 365., 386., 407.],
                         [488., 518., 548., 578.]],
                        [[1100., 1139., 1178., 1217.],
                         [1352., 1400., 1448., 1496.]]],
                       [[[2288., 2345., 2402., 2459.],
                         [2648., 2714., 2780., 2846.]],
                        [[3908., 3983., 4058., 4133.],
                         [4376., 4460., 4544., 4628.]],
                        [[5960., 6053., 6146., 6239.],
                         [6536., 6638., 6740., 6842.]]]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_4d_transpose_a():
    input_x = Tensor(np.arange(2 * 3 * 3 * 2).reshape((2, 3, 3, 2)), mstype.float32)
    input_y = Tensor(np.arange(2 * 3 * 3 * 4).reshape((2, 3, 3, 4)), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = BatchMatMulNet(transpose_a=True)
    output = net(input_x, input_y)
    expect = np.array([[[[40., 46., 52., 58.],
                         [52., 61., 70., 79.]],
                        [[400., 424., 448., 472.],
                         [448., 475., 502., 529.]],
                        [[1192., 1234., 1276., 1318.],
                         [1276., 1321., 1366., 1411.]]],
                       [[[2416., 2476., 2536., 2596.],
                         [2536., 2599., 2662., 2725.]],
                        [[4072., 4150., 4228., 4306.],
                         [4228., 4309., 4390., 4471.]],
                        [[6160., 6256., 6352., 6448.],
                         [6352., 6451., 6550., 6649.]]]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_4d_transpose_b():
    input_x = Tensor(np.arange(2 * 3 * 2 * 3).reshape((2, 3, 2, 3)), mstype.float32)
    input_y = Tensor(np.arange(2 * 3 * 4 * 3).reshape((2, 3, 4, 3)), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = BatchMatMulNet(transpose_b=True)
    output = net(input_x, input_y)
    expect = np.array([[[[5.000e+00, 1.400e+01, 2.300e+01, 3.200e+01],
                         [1.400e+01, 5.000e+01, 8.600e+01, 1.220e+02]],
                        [[2.750e+02, 3.380e+02, 4.010e+02, 4.640e+02],
                         [3.920e+02, 4.820e+02, 5.720e+02, 6.620e+02]],
                        [[9.770e+02, 1.094e+03, 1.211e+03, 1.328e+03],
                         [1.202e+03, 1.346e+03, 1.490e+03, 1.634e+03]]],
                       [[[2.111e+03, 2.282e+03, 2.453e+03, 2.624e+03],
                         [2.444e+03, 2.642e+03, 2.840e+03, 3.038e+03]],
                        [[3.677e+03, 3.902e+03, 4.127e+03, 4.352e+03],
                         [4.118e+03, 4.370e+03, 4.622e+03, 4.874e+03]],
                        [[5.675e+03, 5.954e+03, 6.233e+03, 6.512e+03],
                         [6.224e+03, 6.530e+03, 6.836e+03, 7.142e+03]]]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_4d_transpose_ab():
    input_x = Tensor(np.arange(2 * 3 * 3 * 2).reshape((2, 3, 3, 2)), mstype.float16)
    input_y = Tensor(np.arange(2 * 3 * 4 * 3).reshape((2, 3, 4, 3)), mstype.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = BatchMatMulNet(transpose_a=True, transpose_b=True)
    output = net(input_x, input_y)
    expect = np.array([[[[10., 28., 46., 64.],
                         [13., 40., 67., 94.]],
                        [[316., 388., 460., 532.],
                         [355., 436., 517., 598.]],
                        [[1054., 1180., 1306., 1432.],
                         [1129., 1264., 1399., 1534.]]],
                       [[[2224., 2404., 2584., 2764.],
                         [2336., 2524., 2712., 2902.]],
                        [[3826., 4060., 4296., 4528.],
                         [3974., 4216., 4460., 4704.]],
                        [[5860., 6148., 6436., 6728.],
                         [6040., 6340., 6636., 6932.]]]], np.float16)
    judge_result_correct(output.asnumpy(), expect)


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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bmm_forward_float32_tensor_api():
    """
    Feature: test bmm forward tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_bmm_forward_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bmm_forward_float32_functional_api():
    """
    Feature: test bmm forward functional api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_bmm_forward_functional_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_bmm_forward_functional_api(np.float32)


if __name__ == '__main__':
    test_bmm_forward_float32_tensor_api()
    test_bmm_forward_float32_functional_api()


class BatchMatMul(Cell):
    def __init__(self):
        super().__init__()
        self.batchmatmul = op.matmul

    def construct(self, x1, x2):
        return self.batchmatmul(x1, x2)


class BatchMatMulTestNet(Cell):
    def __init__(self, inputs=None):
        self.ms_type = inputs[0].dtype

        self.input_x1 = inputs[0]
        self.input_x1_np = inputs[0].asnumpy()
        self.input_x1_shape = inputs[0].shape

        self.input_x2 = inputs[1]
        self.input_x2_np = inputs[1].asnumpy()
        self.input_x2_shape = inputs[1].shape
        self.loss = 1e-3

    def forward_mindspore_impl(self):
        input_x1 = Tensor(self.input_x1)
        input_x2 = Tensor(self.input_x2)
        net = BatchMatMul()
        out = net(input_x1, input_x2)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_float16():
    """
    Feature: test bmm with dtype float16.
    Description: test bmm with dtype float16.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_float32():
    """
    Feature: test bmm with dtype float32.
    Description: test bmm with dtype float32.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_float64():
    """
    Feature: test bmm with dtype float64.
    Description: test bmm with dtype float64.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_int8():
    """
    Feature: test bmm with dtype int8.
    Description: test bmm with dtype int8.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.int8)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.int8)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.int8)
    assert (out.asnumpy() == expect).all()
    assert str(out.dtype) == "Int32"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_int16():
    """
    Feature: test bmm with dtype int16.
    Description: test bmm with dtype int16.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.int16)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.int16)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    int16_out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.int16)
    assert (int16_out.asnumpy() == expect).all()
    assert str(int16_out.dtype) == "Int16"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_int32():
    """
    Feature: test bmm with dtype int32.
    Description: test bmm with dtype int32.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.int32)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.int32)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    int32_out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.int32)
    assert (int32_out.asnumpy() == expect).all()
    assert str(int32_out.dtype) == "Int32"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_int64():
    """
    Feature: test bmm with dtype int64.
    Description: test bmm with dtype int64.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.int64)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.int64)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    int64_out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.int64)
    assert (int64_out.asnumpy() == expect).all()
    assert str(int64_out.dtype) == "Int64"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_uint8():
    """
    Feature: test bmm with dtype uint8.
    Description: test bmm with dtype uint8.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.uint8)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.uint8)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    uint8_out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.int8)
    assert (uint8_out.asnumpy() == expect).all()
    assert str(uint8_out.dtype) == "UInt8"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_uint16():
    """
    Feature: test bmm with dtype uint16.
    Description: test bmm with dtype uint16.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.uint16)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.uint16)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    uint16_out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.int16)
    assert (uint16_out.asnumpy() == expect).all()
    assert str(uint16_out.dtype) == "UInt16"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_uint32():
    """
    Feature: test bmm with dtype uint32.
    Description: test bmm with dtype uint32.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.uint32)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.uint32)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    uint32_out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.uint32)
    assert (uint32_out.asnumpy() == expect).all()
    assert str(uint32_out.dtype) == "UInt32"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_uint64():
    """
    Feature: test bmm with dtype uint64.
    Description: test bmm with dtype uint64.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_list = []
    input_x1 = Tensor(np.ones(shape=[4, 1, 3]), ms.uint64)
    input_x2 = Tensor(np.ones(shape=[4, 3, 4]), ms.uint64)
    input_list.append(input_x1)
    input_list.append(input_x2)
    fact = BatchMatMulTestNet(inputs=input_list)
    uint64_out = fact.forward_mindspore_impl()
    expect = np.array([[[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]],
                       [[3, 3, 3, 3]]], np.uint64)
    assert (uint64_out.asnumpy() == expect).all()
    assert str(uint64_out.dtype) == "UInt64"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_complex64():
    """
    Feature: test bmm with dtype complex64.
    Description: test bmm with dtype complex64.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_type_complex128():
    """
    Feature: test bmm with dtype complex128.
    Description: test bmm with dtype complex128.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchmatmul_default_transpose():
    """
    Feature: test bmm with default transpose.
    Description: test bmm with default transpose.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_a = [[[1.73676595], [0.12718279]]]
    input_b = [[[1.04756699, 0.40877772, -0.05073088]]]
    net = BatchMatMulNet()
    output = net(Tensor(input_a), Tensor(input_b))
    expected = [[[1.8193787, 0.7099512, -0.08810767],
                 [0.1332325, 0.05198949, -0.0064521]]]
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
