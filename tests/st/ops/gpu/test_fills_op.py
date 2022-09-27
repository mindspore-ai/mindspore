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
import pytest

from mindspore import context, nn, Tensor
from mindspore import ops as P
from mindspore.ops.operations import _inner_ops as inner


class FillsNet(nn.Cell):
    """FillsNet."""
    def __init__(self):
        super(FillsNet, self).__init__()
        self.fills = P.fills

    def construct(self, x, value):
        out = self.fills(x, value)
        return out


class FillsDynamicNet(nn.Cell):
    """Fills in dynamic shape."""
    def __init__(self):
        super(FillsDynamicNet, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, x, value):
        x = self.test_dynamic(x)
        out = P.fills(x, value)
        return out


def compare_with_numpy(data_shape, data_type, value, out):
    """Compare results with numpy."""
    expect_res = np.zeros(data_shape, dtype=data_type)
    expect_res.fill(value)
    ms_res = out.asnumpy()
    assert np.allclose(expect_res, ms_res)


def gen_np_input(data_shape, data_type):
    """Generate input x."""
    out = np.random.randn(*data_shape)
    if not data_shape:
        out = data_type(out)
    else:
        out = out.astype(data_type)
    return out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_shape', [(), (2,), (2, 3), (2, 2, 3, 3, 4, 4, 5)])
@pytest.mark.parametrize('data_type', [np.int8, np.int16, np.int32, np.float16, np.float32])
def test_fills_data_type_and_shape(run_mode, data_shape, data_type):
    """
    Feature: Fills
    Description:  test cases for Fills operator with multiple data types and shapes.
    Expectation: the result match numpy.
    """
    context.set_context(mode=run_mode, device_target='GPU')
    input_np = gen_np_input(data_shape=data_shape, data_type=data_type)
    input_x = Tensor(input_np)
    value = 4.0
    model = FillsNet()
    out = model(input_x, value)
    compare_with_numpy(data_shape, data_type, value, out)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('value', [4, 4.0, Tensor(np.float32(4))])
def test_fills_with_value_type(run_mode, value):
    """
    Feature: Fills
    Description:  test cases for Fills operator with different value type.
    Expectation: the result match numpy.
    """
    context.set_context(mode=run_mode, device_target='GPU')
    data_shape = (2, 3)
    data_type = np.int32
    input_np = gen_np_input(data_shape=data_shape, data_type=data_type)
    input_x = Tensor(input_np)
    model = FillsNet()
    out = model(input_x, value)
    compare_with_numpy(data_shape, data_type, value, out)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_shape', [(2,), (2, 3), (2, 2, 3, 3, 4, 4, 5)])
def test_fills_dyn_with_dynamic_shape(run_mode, data_shape):
    """
    Feature: Fills
    Description:  test cases for Fills operator in dynamic shape case.
    Expectation: the result match numpy.
    """
    data_type = np.int32
    context.set_context(mode=run_mode, device_target='GPU')
    input_np = gen_np_input(data_shape=data_shape, data_type=data_type)
    input_x = Tensor(input_np)
    value = 4.0
    model = FillsDynamicNet()
    out = model(input_x, value)
    compare_with_numpy(data_shape, data_type, value, out)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
def test_fills_with_nan(run_mode, data_type):
    """
    Feature: Fills
    Description:  test cases for Fills operator when fill with nan.
    Expectation: the result match numpy.
    """
    context.set_context(mode=run_mode, device_target='GPU')
    data_shape = (2, 3)
    value = float('nan')
    input_np = gen_np_input(data_shape=data_shape, data_type=data_type)
    input_x = Tensor(input_np)
    out = input_x.fills(value)
    assert np.isnan(out.asnumpy()).any()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
@pytest.mark.parametrize('value', [float('inf'), float('-inf')])
def test_fills_with_inf(data_type, value):
    """
    Feature: Fills
    Description:  test cases for Fills operator when fill with inf.
    Expectation: the result match numpy.
    """
    context.set_context(device_target='GPU')
    data_shape = (2, 3)
    input_np = gen_np_input(data_shape=data_shape, data_type=data_type)
    input_x = Tensor(input_np)
    out = input_x.fills(value)
    compare_with_numpy(data_shape, data_type, value, out)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_type', [np.int8, np.int16, np.int32, np.float16])
def test_fills_with_overflow(run_mode, data_type):
    """
    Feature: Fills
    Description:  test cases for Fills operator when overflow happens on value convert.
    Expectation: the result match numpy.
    """
    context.set_context(mode=run_mode, device_target='GPU')
    data_shape = (2, 3)
    input_np = gen_np_input(data_shape=data_shape, data_type=data_type)
    input_x = Tensor(input_np)
    value = float(pow(2, 32))
    model = FillsNet()
    with pytest.raises(RuntimeError, match='Fills-op'):
        model(input_x, value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [np.int8, np.int16, np.int32])
@pytest.mark.parametrize('value', [float('inf'), float('-inf'), float('nan')])
def test_fills_except_with_inf_nan(data_type, value):
    """
    Feature: Fills
    Description:  test cases for Fills operator when convert inf/nan to int type.
    Expectation: the result match numpy.
    """
    context.set_context(device_target='GPU')
    data_shape = (2, 3)
    input_np = gen_np_input(data_shape=data_shape, data_type=data_type)
    input_x = Tensor(input_np)
    with pytest.raises(RuntimeError, match='Fills-op'):
        input_x.fills(value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fills_except_with_invalid_type():
    """
    Feature: Fills
    Description:  test cases for Fills operator with invalid type.
    Expectation: the result match numpy.
    """
    context.set_context(device_target='GPU')
    data_shape = (2, 3)
    input_np = gen_np_input(data_shape=data_shape, data_type=np.int)
    input_x = Tensor(input_np)
    value = [2]
    with pytest.raises(TypeError, match='ops.fills'):
        P.fills(input_x, value)
