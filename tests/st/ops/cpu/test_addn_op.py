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
from mindspore import Tensor, ops
from mindspore.ops import operations as P
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net2Inputs(nn.Cell):
    def __init__(self):
        super(Net2Inputs, self).__init__()
        self.addn = P.AddN()

    def construct(self, x, y):
        return self.addn((x, y))


class Net2InputsDynRank(nn.Cell):
    def __init__(self):
        super(Net2InputsDynRank, self).__init__()
        self.op = P.AddN()
        self.reduce_sum = P.ReduceSum(keep_dims=False)

    def construct(self, x, y, dyn_reduce_axis):
        x = self.reduce_sum(x, dyn_reduce_axis)
        y = self.reduce_sum(y, dyn_reduce_axis)
        res = self.op((x, y))
        return res


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_two_tensors_add():
    """
    Feature: ALL To ALL
    Description: test cases for AddN of two tensors
    Expectation: the result match to numpy
    """
    x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    y = np.arange(88, 2 * 3 * 2 + 88).reshape((2, 3, 2))
    addn_net = Net2Inputs()
    dtypes = (np.int32, np.float32, np.float64)
    for dtype in dtypes:
        output = addn_net(Tensor(x.astype(dtype)), Tensor(y.astype(dtype)))
        expect_result = (x + y).astype(dtype)
        assert output.asnumpy().dtype == expect_result.dtype
        assert np.array_equal(output.asnumpy(), expect_result)

    # Test for dynamic shape of addn.
    input_x_dyn = Tensor(shape=[2, None, 2], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, 3, None], dtype=mstype.float32)
    addn_dyn_net = Net2Inputs()
    addn_dyn_net.set_inputs(input_x_dyn, input_y_dyn)
    dyn_output = addn_dyn_net(Tensor(x.astype(np.float32)), Tensor(y.astype(np.float32)))
    expect_dync_result = (x + y).astype(np.float32)
    assert np.array_equal(dyn_output.asnumpy(), expect_dync_result)

    # test dynamic_rank
    dyn_rank_net = Net2InputsDynRank()
    input_x_dyn = Tensor(shape=[2, None, 2, 1], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, 3, None, 1], dtype=mstype.float32)
    dyn_reduce_axis = Tensor(shape=[None], dtype=mstype.int64)
    dyn_rank_net.set_inputs(input_x_dyn, input_y_dyn, dyn_reduce_axis)

    reduce_axis = np.array([-1], dtype=np.int64)
    output = dyn_rank_net(Tensor(np.expand_dims(x, -1), mstype.float32),
                          Tensor(np.expand_dims(y, -1), mstype.float32), Tensor(reduce_axis))
    assert np.array_equal(output.asnumpy(), expect_dync_result)


class Net4Inputs(nn.Cell):
    def __init__(self):
        super(Net4Inputs, self).__init__()
        self.addn = P.AddN()

    def construct(self, x, y, m, n):
        return self.addn((x, y, m, n))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_four_tensors_add():
    """
    Feature: ALL To ALL
    Description: test cases for AddN of four tensors
    Expectation: the result match to numpy
    """
    x = np.arange(2 * 3).reshape((2, 3))
    y = np.arange(1, 2 * 3 + 1).reshape((2, 3))
    m = np.arange(2, 2 * 3 + 2).reshape((2, 3))
    n = np.arange(3, 2 * 3 + 3).reshape((2, 3))
    addn_net = Net4Inputs()
    dtypes = (np.int32, np.float32, np.float64)
    for dtype in dtypes:
        output = addn_net(Tensor(x.astype(dtype)), Tensor(y.astype(dtype)),
                          Tensor(m.astype(dtype)), Tensor(n.astype(dtype)))
        expect_result = (x + y + m + n).astype(dtype)
        assert output.asnumpy().dtype == expect_result.dtype
        assert np.array_equal(output.asnumpy(), expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_addn_support_type():
    """
    Feature: test ops.addn.
    Description: test ops.addn with different types.
    Expectation: the result match with expected result.
    """
    out_fp16 = ops.addn([Tensor([1.5, 2.5, 3.5], mstype.float16), Tensor([4.5, 5.5, 6.5], mstype.float16)])
    out_fp32 = ops.addn([Tensor([1.5, 2.5, 3.5], mstype.float32), Tensor([4.5, 5.5, 6.5], mstype.float32)])
    out_fp64 = ops.addn([Tensor([1.5, 2.5, 3.5], mstype.float64), Tensor([4.5, 5.5, 6.5], mstype.float64)])
    out_int8 = ops.addn([Tensor([1, 2, 3], mstype.int8), Tensor([4, 5, 6], mstype.int8)])
    out_int16 = ops.addn([Tensor([1, 2, 3], mstype.int16), Tensor([4, 5, 6], mstype.int16)])
    out_int32 = ops.addn([Tensor([1, 2, 3], mstype.int32), Tensor([4, 5, 6], mstype.int32)])
    out_int64 = ops.addn([Tensor([1, 2, 3], mstype.int64), Tensor([4, 5, 6], mstype.int64)])
    out_uint8 = ops.addn([Tensor([1, 2, 3], mstype.uint8), Tensor([4, 5, 6], mstype.uint8)])
    out_uint16 = ops.addn([Tensor([1, 2, 3], mstype.uint16), Tensor([4, 5, 6], mstype.uint16)])
    out_uint32 = ops.addn([Tensor([1, 2, 3], mstype.uint32), Tensor([4, 5, 6], mstype.uint32)])
    out_uint64 = ops.addn([Tensor([1, 2, 3], mstype.uint64), Tensor([4, 5, 6], mstype.uint64)])
    out_complex64 = ops.addn([Tensor(np.asarray(np.complex(1.5 + 0.4j)), mstype.complex64),
                              Tensor(np.asarray(np.complex(2.5 + 0.4j)), mstype.complex64)])
    out_complex128 = ops.addn([Tensor(np.asarray(np.complex(1.5 + 0.4j)), mstype.complex128),
                               Tensor(np.asarray(np.complex(2.5 + 0.4j)), mstype.complex128)])

    assert np.allclose(out_fp16.asnumpy(), Tensor([6., 8., 10.], mstype.float16).asnumpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(out_fp32.asnumpy(), Tensor([6., 8., 10.], mstype.float32).asnumpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(out_fp64.asnumpy(), Tensor([6., 8., 10.], mstype.float64).asnumpy(), rtol=1e-5, atol=1e-5)
    assert np.all(out_int8.asnumpy() == Tensor([5, 7, 9], mstype.int8).asnumpy())
    assert np.all(out_int16.asnumpy() == Tensor([5, 7, 9], mstype.int16).asnumpy())
    assert np.all(out_int32.asnumpy() == Tensor([5, 7, 9], mstype.int32).asnumpy())
    assert np.all(out_int64.asnumpy() == Tensor([5, 7, 9], mstype.int64).asnumpy())
    assert np.all(out_uint8.asnumpy() == Tensor([5, 7, 9], mstype.uint8).asnumpy())
    assert np.all(out_uint16.asnumpy() == Tensor([5, 7, 9], mstype.uint16).asnumpy())
    assert np.all(out_uint32.asnumpy() == Tensor([5, 7, 9], mstype.uint32).asnumpy())
    assert np.all(out_uint64.asnumpy() == Tensor([5, 7, 9], mstype.uint64).asnumpy())
    assert np.all(out_complex64.asnumpy() == Tensor(np.asarray(np.complex(4 + 0.8j)), mstype.complex64).asnumpy())
    assert np.all(out_complex128.asnumpy() == Tensor(np.asarray(np.complex(4 + 0.8j)), mstype.complex128).asnumpy())
