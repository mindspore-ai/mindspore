# Copyright 2026 Huawei Technologies Co., Ltd
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
""" test tensor meta device """

import pytest
import numpy as np

import mindspore as ms
from mindspore import mint, nn, Tensor, Parameter
from mindspore.common.initializer import initializer, One
from mindspore._c_expression import DeviceManagerConf
from tests.mark_utils import arg_mark


class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def construct(self, x):
        return self.fc(x)


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_init_on_meta_device(mode):
    """
    Feature: Meta device
    Description: Init tensor with meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    # Tensor()
    t = Tensor([[1,2,3], [4,5,6]], dtype=ms.int32, device='meta')
    assert t.device == 'meta'
    assert t.shape == (2,3)
    assert t.dtype == ms.int32
    assert t.data_ptr() == 0

    t = Tensor(shape=(2,3), dtype=ms.int32, init=One(), device='meta')
    assert t.device == 'meta'
    assert t.shape == (2,3)
    assert t.dtype == ms.int32
    assert t.data_ptr() == 0

    nd = np.array([[1,2,3],[4,5,6]])
    t = Tensor(nd, dtype=ms.float64, device='meta')
    assert t.device == 'meta'
    assert t.shape == (2,3)
    assert t.dtype == ms.float64
    assert t.data_ptr() == 0

    # ms.tensor()
    t = ms.tensor([[1,2,3], [4,5,6]], dtype=ms.int32, device='meta')
    assert t.device == 'meta'
    assert t.shape == (2,3)
    assert t.dtype == ms.int32
    assert t.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_empty_on_meta_device(mode):
    """
    Feature: Meta device
    Description: Create tensor by mint.empty with meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    t = mint.empty((2,3), dtype=ms.int32, device='meta')
    assert t.device == 'meta'
    assert t.shape == (2,3)
    assert t.dtype == ms.int32
    assert t.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_empty_like_on_meta_device(mode):
    """
    Feature: Meta device
    Description: Create tensor by mint.empty_like with meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x = Tensor([[1,2,3], [4,5,6]], dtype=ms.int32)
    t = mint.empty_like(x, device='meta')
    assert t.device == 'meta'
    assert t.shape == (2,3)
    assert t.dtype == ms.int32
    assert t.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_with_device_context_on_meta_device(mode):
    """
    Feature: Meta device
    Description: Test meta device by device context
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    orig_device_target = DeviceManagerConf.get_instance().get_device_target()

    with ms.DeviceCtx('meta'):
        assert DeviceManagerConf.get_instance().get_device_target() == "meta"

        t = Tensor([[1,2,3], [4,5,6]], dtype=ms.int32)
        assert t.device == 'meta'
        assert t.shape == (2,3)
        assert t.dtype == ms.int32
        assert t.data_ptr() == 0

        t = mint.empty((2,3), dtype=ms.int32)
        assert t.device == 'meta'
        assert t.shape == (2,3)
        assert t.dtype == ms.int32
        assert t.data_ptr() == 0

        x = Tensor([[1,2,3], [4,5,6]], dtype=ms.int32)
        t = mint.empty_like(x)
        assert t.device == 'meta'
        assert t.shape == (2,3)
        assert t.dtype == ms.int32
        assert t.data_ptr() == 0

        p = Parameter(initializer(One(), (2,3), ms.int32))
        assert p.device == 'meta'
        assert p.shape == (2,3)
        assert p.dtype == ms.int32
        assert p.data_ptr() == 0

        x = mint.empty((2,3), dtype=ms.int32)
        p = Parameter(x)
        assert p.device == 'meta'
        assert p.shape == (2,3)
        assert p.dtype == ms.int32
        assert p.data_ptr() == 0

        p = Parameter(10)
        assert p.device == 'meta'
        assert p.shape == ()
        assert p.dtype == ms.int32
        assert p.data_ptr() == 0

        net = MyNet()
        for p in net.get_parameters():
            assert p.device == 'meta'
            assert p.data_ptr() == 0
        
        # test forward
        net = MyNet()
        x = mint.rand((3,2), dtype=ms.float32)
        output = net(x)
        assert output.device == 'meta'
        assert output.shape == (3,2)
        assert output.dtype == ms.float32
        assert output.data_ptr() == 0

    assert DeviceManagerConf.get_instance().get_device_target() == orig_device_target


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_is_meta(mode):
    """
    Feature: Tensor.is_meta
    Description: Test tensor.is_meta
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x = Tensor([[1,2,3], [4,5,6]], dtype=ms.int32)
    assert not x.is_meta

    x = Tensor([[1,2,3], [4,5,6]], dtype=ms.int32, device='meta')
    assert x.is_meta


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_meta_tensor_compute(mode):
    """
    Feature: Meta device
    Description: Test compute operator on meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        t = Tensor([[1,1], [1,1], [1,1]], dtype=ms.float32)
        x = mint.rand((2,3), dtype=ms.float32)
        y = mint.rand_like(t, dtype=ms.float32)
        z = mint.matmul(x,y)
        assert z.device == 'meta'
        assert z.shape == (2,2)
        assert z.dtype == ms.float32
        assert z.data_ptr() == 0

        z = x.matmul(y)
        assert z.device == 'meta'
        assert z.shape == (2,2)
        assert z.dtype == ms.float32
        assert z.data_ptr() == 0

        x = mint.rand((2,3), dtype=ms.float32)
        y = mint.rand((2,3), dtype=ms.float32)
        z = x + y
        assert z.device == 'meta'
        assert z.shape == (2,3)
        assert z.dtype == ms.float32
        assert z.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_meta_tensor_transpose(mode):
    """
    Feature: Meta device
    Description: Test view operator on meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        x = mint.rand((2,3), dtype=ms.float32)
        y = mint.transpose(x, 0, 1)
        assert y.device == 'meta'
        assert y.shape == (3,2)
        assert y.dtype == ms.float32
        assert y.data_ptr() == 0

        y = x.t()
        assert y.device == 'meta'
        assert y.shape == (3,2)
        assert y.dtype == ms.float32
        assert y.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_meta_tensor_reshape(mode):
    """
    Feature: Meta device
    Description: Test view operator on meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        x = mint.rand((2,3), dtype=ms.float32)

        y = x.reshape(1,6)
        assert y.device == 'meta'
        assert y.shape == (1,6)
        assert y.dtype == ms.float32
        assert y.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_meta_tensor_view(mode):
    """
    Feature: Meta device
    Description: Test view operator on meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        x = mint.rand((2,3), dtype=ms.float32)

        y = x.view(6,1)
        assert y.device == 'meta'
        assert y.shape == (6,1)
        assert y.dtype == ms.float32
        assert y.data_ptr() == 0

        y = x.view(ms.float16)
        assert y.device == 'meta'
        assert y.shape == (2,6)  # ms.float32 view to ms.float16, bytes amount is same
        assert y.dtype == ms.float16
        assert y.data_ptr() == 0

        y = x.view(ms.int32)
        assert y.device == 'meta'
        assert y.shape == (2,3)  # ms.float32 view to ms.int32, bytes amount is same
        assert y.dtype == ms.int32
        assert y.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_meta_tensor_to(mode):
    """
    Feature: Meta device
    Description: Test tensor.to() operator on meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    original_tensor = Tensor([[1,2,3],[4,5,6]], dtype=ms.float32)

    with ms.DeviceCtx('meta'):
        x = mint.rand((2,3), dtype=ms.float32)
        y = x.to(ms.int32)
        assert y.device == 'meta'
        assert y.shape == (2,3)
        assert y.dtype == ms.int32
        assert y.data_ptr() == 0

        y = x.to('meta')
        assert y.device == 'meta'
        assert y.shape == (2,3)
        assert y.dtype == ms.float32
        assert y.data_ptr() == 0

        with pytest.raises(RuntimeError):
            y = x.to('cpu')
            assert y.device == 'meta'

        y = original_tensor.to('meta')
        assert y.device == 'meta'
        assert y.shape == (2,3)
        assert y.dtype == ms.float32
        assert y.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_meta_tensor_value_dependency_op(mode):
    """
    Feature: Meta device
    Description: Test value dependency operator on meta device
    Expectation: throw exception
    """
    ms.context.set_context(mode=mode)

    mask = Tensor([1,0,1,0])
    with ms.DeviceCtx('meta'):
        x = mint.rand((3,4), dtype=ms.float32)

        with pytest.raises(RuntimeError):
            y = mint.nonzero(x)
            assert y.device == 'meta'
  
        with pytest.raises(RuntimeError):
            y = mint.unique(x)
            assert y.device == 'meta'

        with pytest.raises(RuntimeError):
            y = mint.masked_select(x, mask)
            assert y.device == 'meta'

        with pytest.raises(RuntimeError):
            y = mint.bincount(x)
            assert y.device == 'meta'


@arg_mark(plat_marks=['cpu_linux','platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_meta_tensor_onehot(mode):
    """
    Feature: Meta device
    Description: Test value dependency operator on meta device
    Expectation: throw exception
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        x = Tensor([1,2,3], dtype=ms.int32)

        y = mint.nn.functional.one_hot(x, 6)
        assert y.device == 'meta'
        assert y.shape == (3, 6)
        assert y.dtype == ms.int32
        assert y.data_ptr() == 0

        with pytest.raises(RuntimeError):
            y = mint.nn.functional.one_hot(x) # num_classes default -1, infer shape is dynamic
            assert y.device == 'meta'
