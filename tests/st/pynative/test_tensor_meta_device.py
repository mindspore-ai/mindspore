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


def test_meta_tensor_random_ops(mode):
    """
    Feature: Meta device
    Description: Test random tensor creation operators on meta device
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        x = mint.randn((2, 3), dtype=ms.float32)
        assert x.device == 'meta'
        assert x.shape == (2, 3)
        assert x.dtype == ms.float32
        assert x.data_ptr() == 0

        t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float32)
        y = mint.randn_like(t)
        assert y.device == 'meta'
        assert y.shape == (2, 2)
        assert y.dtype == ms.float32
        assert y.data_ptr() == 0

        z = mint.randint(0, 10, (3, 4), dtype=ms.int32)
        assert z.device == 'meta'
        assert z.shape == (3, 4)
        assert z.dtype == ms.int32
        assert z.data_ptr() == 0

        w = mint.randint_like(t, 0, 10, dtype=ms.int32)
        assert w.device == 'meta'
        assert w.shape == (2, 2)
        assert w.dtype == ms.int32
        assert w.data_ptr() == 0


def test_meta_tensor_nn_conv2d(mode):
    """
    Feature: Meta device
    Description: Test nn.Conv2d on meta device via RunOpStub path
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        conv = nn.Conv2d(3, 16, kernel_size=3, pad_mode='pad', padding=1)
        x = Tensor(np.zeros((1, 3, 8, 8), dtype=np.float32))
        out = conv(x)
        assert out.device == 'meta'
        assert out.shape == (1, 16, 8, 8)
        assert out.dtype == ms.float32
        assert out.data_ptr() == 0


def test_meta_tensor_parameter(mode):
    """
    Feature: Meta device
    Description: Test Parameter creation from a meta tensor (issue #42223)
    Expectation: success, parameter is meta
    """
    import mindspore.ops.operations as P

    ms.context.set_context(mode=mode)

    weight = mint.empty((32, 32), device='meta')
    assert weight.is_meta

    param = Parameter(weight, name="weight")
    assert param.is_meta
    assert param.shape == (32, 32)
    assert param.dtype == ms.float32
    assert param.data_ptr() == 0

    # Test the full scenario from issue #42223
    class MatMulNet(nn.Cell):
        def __init__(self, matmul_shape):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.weight_1 = mint.empty(matmul_shape, device="meta")
            self.weight_2 = mint.empty(matmul_shape, device="meta")
            self.matmul1_weight = Parameter(self.weight_1, name="weight1")
            self.matmul2_weight = Parameter(self.weight_2, name="weight2")

        def construct(self, inputs):
            x = self.matmul1(inputs, self.matmul1_weight)
            x = self.matmul2(x, self.matmul2_weight)
            return x

    net = MatMulNet((32, 32))
    assert net.weight_1.is_meta
    assert net.weight_2.is_meta
    assert net.matmul1_weight.is_meta
    assert net.matmul2_weight.is_meta


def test_meta_tensor_mixed_device_input(mode):
    """
    Feature: Meta device
    Description: Real device tensor and meta tensor can coexist as op inputs on meta device
    Expectation: success, output is meta tensor
    """
    ms.context.set_context(mode=mode)

    real_tensor = Tensor([[1, 2], [3, 4]], dtype=ms.float32)
    assert not real_tensor.is_meta

    with ms.DeviceCtx('meta'):
        meta_tensor = mint.rand((2, 2), dtype=ms.float32)
        assert meta_tensor.is_meta

        result = real_tensor + meta_tensor
        assert result.device == 'meta'
        assert result.shape == (2, 2)
        assert result.dtype == ms.float32
        assert result.data_ptr() == 0

        result2 = mint.matmul(real_tensor, meta_tensor)
        assert result2.device == 'meta'
        assert result2.shape == (2, 2)
        assert result2.dtype == ms.float32
        assert result2.data_ptr() == 0


def test_meta_tensor_data_access_raises(mode):
    """
    Feature: Meta device
    Description: Accessing data of a meta tensor (copy to host) should raise RuntimeError
    Expectation: RuntimeError is raised
    """
    ms.context.set_context(mode=mode)

    meta_tensor = Tensor([[1, 2], [3, 4]], dtype=ms.float32, device='meta')
    assert meta_tensor.is_meta

    with pytest.raises(RuntimeError):
        meta_tensor.asnumpy()

    meta_empty = mint.empty((3, 4), dtype=ms.float32, device='meta')
    assert meta_empty.is_meta

    with pytest.raises(RuntimeError):
        meta_empty.asnumpy()


def test_meta_device_graph_mode_raises(mode):
    """
    Feature: Meta device
    Description: Graph mode compilation under meta device context should raise RuntimeError
    Expectation: RuntimeError is raised
    """
    ms.context.set_context(mode=ms.GRAPH_MODE)

    net = MyNet()
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=ms.float32)

    with ms.DeviceCtx('meta'):
        with pytest.raises(RuntimeError):
            net(x)

    ms.context.set_context(mode=mode)


def test_meta_parameter_init_data(mode):
    """
    Feature: Meta device
    Description: Parameter.init_data() on a meta parameter should return self without initializing data
    Expectation: success, parameter remains meta
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        p = Parameter(initializer(One(), (4, 4), ms.float32), name="meta_param")
        assert p.is_meta
        assert p.has_init

        result = p.init_data()
        assert result is p
        assert result.is_meta
        assert result.shape == (4, 4)
        assert result.dtype == ms.float32
        assert result.data_ptr() == 0


def test_meta_tensor_contiguous(mode):
    """
    Feature: Meta device
    Description: Operations requiring contiguous tensors on meta device should not crash
    Expectation: success, output is meta tensor
    """
    ms.context.set_context(mode=mode)

    with ms.DeviceCtx('meta'):
        x = mint.rand((3, 4, 5), dtype=ms.float32)
        y = mint.transpose(x, 0, 2)
        assert y.device == 'meta'
        assert y.shape == (5, 4, 3)

        z = y.reshape(60)
        assert z.device == 'meta'
        assert z.shape == (60,)
        assert z.data_ptr() == 0

        w = mint.rand((2, 3), dtype=ms.float32)
        wt = w.t()
        assert wt.device == 'meta'
        assert wt.shape == (3, 2)

        v = wt.reshape(6)
        assert v.device == 'meta'
        assert v.shape == (6,)
        assert v.data_ptr() == 0


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_all(mode):
    """
    Feature: Meta device
    Description: Test all meta device scenarios
    Expectation: success
    """
    test_tensor_init_on_meta_device(mode)
    test_empty_on_meta_device(mode)
    test_empty_like_on_meta_device(mode)
    test_with_device_context_on_meta_device(mode)
    test_is_meta(mode)
    test_meta_tensor_compute(mode)
    test_meta_tensor_transpose(mode)
    test_meta_tensor_reshape(mode)
    test_meta_tensor_view(mode)
    test_meta_tensor_to(mode)
    test_meta_tensor_value_dependency_op(mode)
    test_meta_tensor_onehot(mode)
    test_meta_tensor_random_ops(mode)
    test_meta_tensor_nn_conv2d(mode)
    test_meta_tensor_parameter(mode)
    test_meta_tensor_mixed_device_input(mode)
    test_meta_tensor_data_access_raises(mode)
    test_meta_device_graph_mode_raises(mode)
    test_meta_parameter_init_data(mode)
    test_meta_tensor_contiguous(mode)
