# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test hyper offload Ops CopyToHost"""
import numpy as np
import mindspore as ms
from mindspore import jit, ops
from mindspore import Tensor
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_remote_ops_copy_to_host():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    @jit
    def foo(x):
        x = ops.auto_generate.CopyToHost()(x)
        return x
    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    assert ret.device == "CPU"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_remote_ops_copy_to_host_and_element_wise():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    @jit
    def foo(x,y):
        mul = ops.Mul()
        m = mul(x, y)
        res = ops.auto_generate.CopyToHost()(m)
        return res

    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    input_y = Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    ret = foo(input_x,input_y)
    assert np.all(ret.asnumpy() == np.array((4, 10, 18)))
    assert ret.device == "CPU"
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_remote_ops_copy_to_host_and_squeeze():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    @jit
    def foo(x):
        squeeze = ops.Squeeze(2)
        s = squeeze(x)
        res = ops.auto_generate.CopyToHost()(s)
        return res
    input_tensor = Tensor(np.ones(shape=[3, 2, 1]), ms.float32)
    ret = foo(input_tensor)
    assert np.all(ret.asnumpy() == np.array([[1, 1],[1, 1],[1, 1]]))
    assert ret.device == "CPU"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_remote_ops_copy_to_host_and_broadcast_to():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    @jit
    def foo(x,shape):
        broadcast_to = ops.BroadcastTo(shape)
        broadcast_to(x)
        res = ops.auto_generate.CopyToHost()(x)
        return res
    shape = (2, 3)
    input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    ret = foo(input_x,shape)
    assert np.all(ret.asnumpy() == np.array([[1, 2, 3],[1, 2, 3]]))
    assert ret.device == "CPU"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_remote_ops_copy_to_host_and_smooth_l1_loss():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    @jit
    def foo(input_data, target_data):
        loss = ops.SmoothL1Loss()
        y = loss(input_data, target_data)
        res = ops.auto_generate.CopyToHost()(y)
        return res
    input_data = Tensor(np.array([1, 2, 3]), ms.float32)
    target_data = Tensor(np.array([1, 2, 2]), ms.float32)
    ret = foo(input_data, target_data)
    assert np.all(ret.asnumpy() == np.array([0, 0, 0.5]))
    assert ret.device == "CPU"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_remote_ops_copy_to_host_and_add():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    @jit
    def foo(x,y):
        add = ops.Add()
        z = add(x,y)
        res = ops.auto_generate.CopyToHost()(z)
        return res
    x = Tensor([1, 2, 3, 4])
    y = Tensor([1, 2, 3, 4])
    ret = foo(x,y)
    assert np.all(ret.asnumpy() == np.array((2, 4, 6, 8)))
    assert ret.device == "CPU"
