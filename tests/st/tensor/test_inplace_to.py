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
"""
test tensor inplace to
"""
import os
import pytest
import numpy as np
from tests.mark_utils import arg_mark
from mindspore import nn, Tensor, jit
from mindspore.common.parameter import Parameter
import mindspore as ms


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"
    yield
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("input_x", [Tensor([1, 2, 3, 4]), Parameter(Tensor([1, 2, 3, 4]), name='param')])
def test_tensor_get_data(input_x):
    """
    Feature: Tensor.data
    Description: Test Tensor.data getter with sharing data.
    Expectation: success
    """

    @jit
    def foo(x):
        y = x.data
        return x, y

    ret0, ret1 = foo(input_x)
    assert ret0._data_ptr() == ret1._data_ptr()  # pylint:disable=protected-access
    assert np.all(ret0.asnumpy() == np.array((1, 2, 3, 4)))
    assert np.all(ret1.asnumpy() == np.array((1, 2, 3, 4)))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("input_x", [Tensor([1, 2, 3, 4]), Parameter(Tensor([1, 2, 3, 4]), name='param')])
def test_tensor_get_data_inplace(input_x):
    """
    Feature: Tensor.data
    Description: Test Tensor.data getter with inplace.
    Expectation: success
    """

    @jit
    def foo(x):
        y = x.data
        x.add_(1)
        y.add_(1)
        return x, y

    ret0, ret1 = foo(input_x)
    assert np.all(input_x.asnumpy() == np.array((3, 4, 5, 6)))
    assert ret0._data_ptr() == ret1._data_ptr()  # pylint:disable=protected-access
    assert np.all(input_x.asnumpy() == ret0.asnumpy())
    assert np.all(input_x.asnumpy() == ret1.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize("input_x", [Tensor([1, 2, 3, 4]), Parameter(Tensor([1, 2, 3, 4]), name='param')])
def test_tensor_set_data(input_x):
    """
    Feature: Tensor.data
    Description: Test Tensor.data setter with sharing data.
    Expectation: success
    """

    @jit
    def foo(x, y):
        x.data = y
        return x, y

    input_y = Tensor([5, 6, 7, 8])
    ret0, ret1 = foo(input_x, input_y)
    assert np.all(input_x.asnumpy() == np.array((5, 6, 7, 8)))
    assert ret0._data_ptr() == ret1._data_ptr()  # pylint:disable=protected-access
    assert np.all(input_x.asnumpy() == ret0.asnumpy())
    assert np.all(input_x.asnumpy() == ret1.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize("input_x", [Tensor([1, 2, 3, 4]), Parameter(Tensor([1, 2, 3, 4]), name='param')])
def test_tensor_set_data_inplace(input_x):
    """
    Feature: Tensor.data
    Description: Test Tensor.data setter with inplace.
    Expectation: success
    """

    @jit
    def foo(x, y):
        x.data = y
        x.add_(1)
        y.add_(1)
        return x, y

    input_y = Tensor([5, 6, 7, 8])
    ret0, ret1 = foo(input_x, input_y)
    assert np.all(input_x.asnumpy() == np.array((7, 8, 9, 10)))
    assert np.all(input_x.asnumpy() == input_y.asnumpy())
    assert ret0._data_ptr() == ret1._data_ptr()  # pylint:disable=protected-access
    assert np.all(input_x.asnumpy() == ret0.asnumpy())
    assert np.all(input_x.asnumpy() == ret1.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.skip(reason="Error messages are inconsistent when the value of non_blocking is different.")
@pytest.mark.parametrize("input_x", [Tensor([1, 2, 3, 4]), Parameter(Tensor([1, 2, 3, 4]), name='param')])
@pytest.mark.parametrize("non_blocking", [True, False])
def test_tensor_data_del(input_x, non_blocking):
    """
    Feature: Tensor.data
    Description: Test Tensor.data delete.
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x):
            out = x.data
            x.data.delete_(non_blocking)
            return out

    x0 = Tensor([1, 2, 3, 4])
    pynative_ret = Net()(x0)  # pylint: disable=W0612

    with pytest.raises(RuntimeError) as err:
        net = Net()
        net.construct = jit(net.construct, backend='ms_backend')
        ret1 = net(input_x)  # pylint: disable=W0612
    assert "Async device to device failed" in str(err.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("input_x", [Tensor([1, 2, 3, 4]), Parameter(Tensor([1, 2, 3, 4]), name='param')])
@pytest.mark.parametrize("device_type", ["Ascend", "CPU"])
@pytest.mark.parametrize("non_blocking", [True, False])
def test_tensor_explicit_inplace_to(mode, input_x, device_type, non_blocking):
    """
    Feature: Tensor.data in-place modification
    Description: Test Tensor.data in-place modification.
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, device_type, non_blocking):
            y = x.to(device=device_type, non_blocking=non_blocking, copy=True)
            x.data.delete_(non_blocking)
            x.data = y
            return x, y

    ms.set_context(mode=mode)
    net = Net()
    ret0, ret1 = net(input_x, device_type, non_blocking)
    assert device_type in input_x.device
    assert device_type in ret0.device
    assert device_type in ret1.device
    assert ret0._data_ptr() == ret1._data_ptr()  # pylint:disable=protected-access
    assert np.allclose(input_x.asnumpy(), ret1.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("input_x", [Tensor([1, 2, 3, 4]), Parameter(Tensor([1, 2, 3, 4]), name='param')])
@pytest.mark.parametrize("device_type", ["Ascend", "CPU"])
@pytest.mark.parametrize("non_blocking", [True, False])
def test_tensor_inplace_to(mode, input_x, device_type, non_blocking):
    """
    Feature: Tensor.data in-place modification
    Description: Test Tensor.data in-place modification.
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, device_type, non_blocking):
            y = x.to_(device=device_type, non_blocking=non_blocking)
            return x, y

    ms.set_context(mode=mode)
    net = Net()
    ret0, ret1 = net(input_x, device_type, non_blocking)
    assert device_type in input_x.device
    assert device_type in ret0.device
    assert device_type in ret1.device
    assert ret0._data_ptr() == ret1._data_ptr()  # pylint:disable=protected-access
    assert np.allclose(input_x.asnumpy(), ret1.asnumpy())
