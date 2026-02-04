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
"""Test hyper offload Ops"""

import os
import numpy as np
import mindspore as ms
from mindspore import mutable
from mindspore import jit, ops, context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.common.initializer import One

from tests.mark_utils import arg_mark

os.environ['DEVICE_ID'] = '0'
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=0)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_hiar_addr_to_remote():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        x = ops.auto_generate.UpdateToRemote()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    assert ret.device == "Ascend:0"

    x = Tensor(shape = (6), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6)))
    assert ret.device == "Ascend:0"

    x = Tensor(shape = (6,7), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6,7)))
    assert ret.device == "Ascend:0"

    x = Tensor(shape = (6,7,8), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6,7,8)))
    assert ret.device == "Ascend:0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_hiar_addr_update_to_device():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_HYPER_OFFLOAD'] = '1'

    @jit
    def foo(x):
        x = ops.auto_generate.UpdateToDevice()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))

    x = Tensor(shape = (6), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6)))
    assert ret.device == "Ascend:0"

    x = Tensor(shape = (6,7), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6,7)))
    assert ret.device == "Ascend:0"

    x = Tensor(shape = (6,7,8), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6,7,8)))
    assert ret.device == "Ascend:0"

    os.environ['MS_DEV_HYPER_OFFLOAD'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_hiar_addr_detach():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_HYPER_OFFLOAD'] = '1'

    @jit
    def foo(x):
        x = ops.auto_generate.UpdateToRemote()(x)
        x = ops.auto_generate.Detach()(x)
        x = ops.auto_generate.Detach()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))

    # 不同维度
    x = Tensor(shape = (6), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6)))

    x = Tensor(shape = (6,7), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6,7)))

    x = Tensor(shape = (6,7,8), dtype=ms.float32, init=One())
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.ones((6,7,8)))

    os.environ['MS_DEV_HYPER_OFFLOAD'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_hiar_addr_to_remote_and_detach():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_HYPER_OFFLOAD'] = '1'

    @jit
    def foo(x):
        ops.auto_generate.UpdateToRemote()(x)
        ops.auto_generate.Detach()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    os.environ['MS_DEV_HYPER_OFFLOAD'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_hiar_addr_in_for_variable_loop():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_HYPER_OFFLOAD'] = '1'

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_a")
            self.param_b = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_b")
            self.param_c = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_c")
            self.params = self.trainable_params()
            self.prefetch = ops.auto_generate.UpdateToDevice()
            self.depend = ops.Depend()
            self.detach = ops.auto_generate.Detach()

        @jit
        def construct(self, a, b, c, d):
            m = (a, b, c)
            a = Tensor([0, 0, 0])
            for i in range(d):
                prefetch_result = self.prefetch(self.params[i], sync=False)
                cur = self.depend(m[i], prefetch_result)
                a = a + cur + self.params[i]
                detach_result = self.detach(m[i], sync=False)
                a = self.depend(a, detach_result)
            return a

    x = Tensor([1, 1, 1], dtype=ms.int32)
    y = Tensor([1, 1, 1], dtype=ms.int32)
    z = Tensor([1, 1, 1], dtype=ms.int32)
    d = mutable(3)
    net = Net()
    ret = net(x, y, z, d)
    assert np.all(ret.asnumpy() == np.array((6, 6, 6)))
    os.environ['MS_DEV_HYPER_OFFLOAD'] = '1'
