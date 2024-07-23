# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class NetUniqueConsecutive(nn.Cell):
    def __init__(self, return_idx=False, return_counts=False, axis=None):
        super(NetUniqueConsecutive, self).__init__()
        self.return_idx = return_idx
        self.return_counts = return_counts
        self.axis = axis

    def construct(self, x):
        return ops.unique_consecutive(x, self.return_idx, self.return_counts, self.axis)


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32))
    net = NetUniqueConsecutive()
    out = net(x)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_return_idx():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator that returns idx.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32))
    net = NetUniqueConsecutive(return_idx=True)
    out, idx = net(x)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    exp_idx = np.array([0, 0, 1, 1, 2, 3, 3, 4]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_return_counts():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator that returns counts.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32))
    net = NetUniqueConsecutive(return_counts=True)
    out, counts = net(x)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    exp_counts = np.array([2, 2, 1, 2, 1]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_set_axis_0():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with axis.
    Expectation: No exception.
    """
    x = Tensor(np.array([[[1, 2, 3], [3, 2, 4]], [[1, 2, 3], [3, 2, 4]]]).astype(np.int32))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=0)
    out, idx, counts = net(x)
    exp_out = np.array([[[1, 2, 3], [3, 2, 4]]]).astype(np.int32)
    exp_idx = np.array([0, 0]).astype(np.int32)
    exp_counts = np.array([2]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_set_axis_1():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with axis.
    Expectation: No exception.
    """
    x = Tensor(np.array([[[1, 2, 3], [3, 2, 4]], [[1, 2, 3], [3, 2, 4]]]).astype(np.int32))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=1)
    out, idx, counts = net(x)
    exp_out = np.array([[[1, 2, 3], [3, 2, 4]], [[1, 2, 3], [3, 2, 4]]]).astype(np.int32)
    exp_idx = np.array([0, 1]).astype(np.int32)
    exp_counts = np.array([1, 1]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_1d_int32():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with int32 data.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 2, 3, 3, 1, 2, 2]).astype(np.int32))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=0)
    out, idx, counts = net(x)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    exp_idx = np.array([0, 1, 2, 2, 3, 4, 4]).astype(np.int32)
    exp_counts = np.array([1, 1, 2, 1, 2]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_1d_int64():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with int64 data.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 2, 3, 3, 1, 2, 2]).astype(np.int64))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=0)
    out, idx, counts = net(x)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int64)
    exp_idx = np.array([0, 1, 2, 2, 3, 4, 4]).astype(np.int64)
    exp_counts = np.array([1, 1, 2, 1, 2]).astype(np.int64)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_1d_half():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with half data.
    Expectation: No exception.
    """
    x = Tensor(np.array([0.4, 0.5, 2.2, 2.2, 12.43, 12.43, 0.4, 0.5]).astype(np.float16))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=0)
    out, idx, counts = net(x)
    exp_out = np.array([0.4, 0.5, 2.2, 12.43, 0.4, 0.5]).astype(np.float16)
    exp_idx = np.array([0, 1, 2, 2, 3, 3, 4, 5]).astype(np.int32)
    exp_counts = np.array([1, 1, 2, 2, 1, 1]).astype(np.int32)
    assert np.allclose(out.asnumpy(), exp_out, rtol=1.e-5, atol=1.e-6)
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_1d_float():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with float data.
    Expectation: No exception.
    """
    x = Tensor(np.array([0.5, 0.5, 1.2, 1.3, 6.5, 1.2, 0.5]).astype(np.float32))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=0)
    out, idx, counts = net(x)
    exp_out = np.array([0.5, 1.2, 1.3, 6.5, 1.2, 0.5]).astype(np.float32)
    exp_idx = np.array([0, 0, 1, 2, 3, 4, 5]).astype(np.int32)
    exp_counts = np.array([2, 1, 1, 1, 1, 1]).astype(np.int32)
    assert np.allclose(out.asnumpy(), exp_out, rtol=1.e-5, atol=1.e-6)
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_0d():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with 0-dimensional data.
    Expectation: No exception.
    """
    x = Tensor(5)
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=None)
    out, idx, counts = net(x)
    exp_out = np.array([5])
    exp_idx = 0
    exp_counts = np.array([1])
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_3d():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with 3-dimensional data.
    Expectation: No exception.
    """
    x = Tensor(np.array([[[1, 2, 3], [3, 2, 4], [3, 2, 4], [1, 2, 3]], \
        [[1, 2, 3], [3, 2, 4], [3, 2, 4], [1, 2, 3]]]).astype(np.int32))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=0)
    out, idx, counts = net(x)
    exp_out = np.array([[[1, 2, 3], [3, 2, 4], [3, 2, 4], [1, 2, 3]]]).astype(np.int32)
    exp_idx = np.array([0, 0]).astype(np.int32)
    exp_counts = np.array([2]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_consecutive_3d_axis():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive operator with 3-dimensional data.
    Expectation: No exception.
    """
    x = Tensor(np.array([[[1, 2, 3], [3, 2, 4], [3, 2, 4], [1, 2, 3]], \
        [[1, 2, 3], [3, 2, 4], [3, 2, 4], [1, 2, 3]]]).astype(np.int32))
    net = NetUniqueConsecutive(return_idx=True, return_counts=True, axis=1)
    out, idx, counts = net(x)
    exp_out = np.array([[[1, 2, 3], [3, 2, 4], [1, 2, 3]], \
        [[1, 2, 3], [3, 2, 4], [1, 2, 3]]]).astype(np.int32)
    exp_idx = np.array([0, 1, 1, 2]).astype(np.int32)
    exp_counts = np.array([1, 2, 1]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()


class NetTensor(nn.Cell):
    def construct(self, x, return_idx, return_counts, axis):
        return x.unique_consecutive(return_idx, return_counts, axis)


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_return_output():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive tensor api that only return output.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32))
    net = NetTensor()
    out = net(x, False, False, 0)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_return_idx():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive tensor api that only return output.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32))
    net = NetTensor()
    out, idx = net(x, True, False, 0)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    exp_idx = np.array([0, 0, 1, 1, 2, 3, 3, 4]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_return_counts():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive tensor api that only return output.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32))
    net = NetTensor()
    out, counts = net(x, False, True, 0)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    exp_counts = np.array([2, 2, 1, 2, 1]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (counts.asnumpy() == exp_counts).all()


#@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_return_all():
    """
    Feature: UniqueConsecutive operator.
    Description: Test UniqueConsecutive tensor api that return all.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32))
    net = NetTensor()
    out, idx, counts = net(x, True, True, 0)
    exp_out = np.array([1, 2, 3, 1, 2]).astype(np.int32)
    exp_idx = np.array([0, 0, 1, 1, 2, 3, 3, 4]).astype(np.int32)
    exp_counts = np.array([2, 2, 1, 2, 1]).astype(np.int32)
    assert (out.asnumpy() == exp_out).all()
    assert (idx.asnumpy() == exp_idx).all()
    assert (counts.asnumpy() == exp_counts).all()
