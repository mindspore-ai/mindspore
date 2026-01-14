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

import os
import pytest
import numpy as np
import mindspore as ms
from mindspore import jit, ops
from mindspore import Tensor
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_ops_get_data():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"
    @jit
    def foo(x):
        y = ops.auto_generate.GetData()(x)
        return y

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))

    @jit
    def foo1(x):
        y = ops.auto_generate.GetData()(x)
        x.add_(1)
        return y

    x = Tensor([1, 2, 3, 4])
    ret = foo1(x)
    assert np.all(ret.asnumpy() == np.array((2, 3, 4, 5)))

    @jit
    def foo2(x):
        y = ops.auto_generate.GetData()(x)
        x.add_(1)
        y.add_(1)
        return y

    x = Tensor([1, 2, 3, 4])
    ret = foo2(x)
    assert np.all(ret.asnumpy() == np.array((3, 4, 5, 6)))
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
def test_remote_ops_copy_to_host():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"

    @jit
    def foo(x):
        x = ops.auto_generate.CopyToHost()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    assert ret.device == "CPU"
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_copy_to_host():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: Throw Exception.
    """

    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"

    @jit
    def foo(x):
        x = ops.auto_generate.CopyToHost()(x)
        x.add_(1)
        return x

    with pytest.raises(RuntimeError) as err:
        x = Tensor([1, 2, 3, 4])
        ret = foo(x)
        assert np.all(ret.asnumpy() == np.array((2, 3, 4, 5)))
    assert "Invalid input device tensor type for kernel" in str(err.value)
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_memory_ops_set_data():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"

    @jit
    def foo1(x, a):
        ops.auto_generate.SetData()(x, a)
        return x

    x = Tensor([1, 2, 3, 4])
    a = Tensor([5, 6, 7, 8])
    ret = foo1(x, a)
    assert np.all(ret.asnumpy() == np.array((5, 6, 7, 8)))

    @jit
    def foo2(x, a):
        ops.auto_generate.SetData()(x, a)
        a.add_(1)
        return x

    x = Tensor([1, 2, 3, 4])
    a = Tensor([5, 6, 7, 8])
    ret = foo2(x, a)
    assert np.all(ret.asnumpy() == np.array((6, 7, 8, 9)))

    @jit
    def foo3(x, a):
        ops.auto_generate.SetData()(x, a)
        a.add_(1)
        x.add_(1)
        return a

    x = Tensor([1, 2, 3, 4])
    a = Tensor([5, 6, 7, 8])
    ret = foo3(x, a)
    assert np.all(ret.asnumpy() == np.array((7, 8, 9, 10)))

    @jit
    def foo4(x, a):
        b = a + 1
        y = x * 5
        ops.auto_generate.SetData()(y, b)
        y.add_(1)
        b.mul_(2)
        return y

    x = Tensor([1, 1, 1])
    a = Tensor([2, 2, 2])
    ret = foo4(x, a)
    assert np.all(ret.asnumpy() == np.array((8, 8, 8)))
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_remote_ops_copy_to():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"
    ms.set_context(device_id=0)

    @jit
    def foo(x):
        x = ops.auto_generate.CopyToHost()(x)
        x = ops.auto_generate.CopyToDevice()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    assert ret.device == "Ascend:0"
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_ops_to_host_free_set_data():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"
    ms.set_context(device_id=0)

    @jit
    def foo(x):
        y = ops.auto_generate.CopyToHost()(x)
        z = ops.auto_generate.Free()(x)
        ops.auto_generate.SetData()(z, y)
        return z

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
def test_remote_ops_copy_to_and_free():
    """
    Feature: Hyper offload base operator
    Description: Base scene.
    Expectation: Throw RuntimeError.
    """
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"
    ms.set_context(device_id=0)

    @jit
    def foo(x):
        y = ops.auto_generate.CopyToHost()(x)
        ops.auto_generate.Free()(x)
        z = ops.auto_generate.CopyToHost()(x)
        return y, z

    with pytest.raises(RuntimeError) as err:
        x = Tensor([1, 2, 3, 4])
        y, z = foo(x)
        print(f'y: {y}, device of z: {z.device}')
    assert "The first input of Primitive 'CopyToHost' has been released before." in str(err.value)
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"
