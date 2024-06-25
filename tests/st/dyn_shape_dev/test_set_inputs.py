# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from mindspore import context
import mindspore as ms


class Net(ms.nn.Cell):
    def construct(self, x, y, z):
        add1 = ms.ops.add(x, y)
        return ms.ops.add(add1, z)


class Net1(ms.nn.Cell):
    def construct(self, x, y, z):
        mul_res = ms.ops.mul(x, 2)
        add_res = ms.ops.add(y, z)
        return mul_res, add_res


class NetTuple(ms.nn.Cell):
    def construct(self, tensors, axis):
        return ms.ops.cat(tensors, axis)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dyn_case(mode):
    """
    Feature: fullmode or incremental mode.
    Description: Test fullmode or incremental mode in normal case.
    Expectation: the result match with static result.
    """
    ms.set_context(mode=mode)

    x = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    y = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    z = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    x_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    z_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)

    net_s = Net()
    out_s = net_s(x, y, z)

    net_d1 = Net()
    net_d1.set_inputs(x_dyn, y_dyn, z_dyn)
    out_d1 = net_d1(x, y, z)
    assert np.allclose(out_s.asnumpy(), out_d1.asnumpy())

    net_d2 = Net()
    net_d2.set_inputs(y=y_dyn)
    out_d2 = net_d2(x, y, z)
    assert np.allclose(out_s.asnumpy(), out_d2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dyn_tuple_case(mode):
    """
    Feature: tuple dynamic input case.
    Description: Test tuple input case.
    Expectation: the result match with static result.
    """
    ms.set_context(mode=mode)

    axis = 1

    x1 = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    x2 = ms.Tensor(np.random.random((2, 5, 4)).astype(np.float32))
    x1_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)

    net_s = NetTuple()
    out_s = net_s((x1, x2), axis)

    try:
        net_d1 = NetTuple()
        net_d1.set_inputs(ms.mutable((x1_dyn, x2_dyn)), axis)
        net_d1((x1, x2), axis)
        assert False, "should be a exception"
    except Exception:  # pylint: disable=broad-except
        pass

    try:
        net_d2 = NetTuple()
        net_d2.set_inputs((x1_dyn, x2_dyn), axis)
        net_d2(ms.mutable(x1, x2), axis)
        assert False, "should be a exception"
    except Exception:  # pylint: disable=broad-except
        pass

    try:
        net_d2 = NetTuple()
        net_d2.set_inputs((x1_dyn, x2_dyn), axis)
        net_d2((x1, x2), axis)
        assert False, "should be a exception"
    except Exception:  # pylint: disable=broad-except
        pass

    net_d3 = NetTuple()
    net_d3.set_inputs(ms.mutable((x1_dyn, x2_dyn)), axis)
    out_d3 = net_d3(ms.mutable([x1, x2]), axis)
    assert np.allclose(out_s.asnumpy(), out_d3.asnumpy())

    net_d4 = NetTuple()
    net_d4.set_inputs(ms.mutable([x1_dyn, x2_dyn]), axis)
    out_d4 = net_d4(ms.mutable([x1, x2]), axis)
    assert np.allclose(out_s.asnumpy(), out_d4.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_dyn_func_case(mode):
    """
    Feature: input_signature in jit.
    Description: Test input_signature in jit.
    Expectation: the result match with static result.
    """
    ms.set_context(mode=mode)

    x = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    y = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    z = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    x_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    z_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)

    def func_s(x, y, z):
        add1 = ms.ops.add(x, y)
        return ms.ops.mul(add1, z)

    @ms.jit(input_signature=(x_dyn, y_dyn, z_dyn))
    def func_d1(x, y, z):
        add1 = ms.ops.add(x, y)
        return ms.ops.mul(add1, z)

    @ms.jit(input_signature={"y": y_dyn})
    def func_d2(x, y, z):
        add1 = ms.ops.add(x, y)
        return ms.ops.mul(add1, z)

    out_s = func_s(x, y, z)

    out_d1 = func_d1(x, y, z)
    assert np.allclose(out_s.asnumpy(), out_d1.asnumpy())

    out_d2 = func_d2(x, y, z)
    assert np.allclose(out_s.asnumpy(), out_d2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dyn_mul_case(mode):
    """
    Feature: call more than once set_inputs.
    Description: Test call more than once set_inputs.
    Expectation: the result match with static result.
    """
    ms.set_context(mode=mode)

    x1 = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    y1 = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    z1 = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    x1_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    y1_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    z1_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)

    x2 = ms.Tensor(np.random.random((2, 4)).astype(np.float32))
    y2 = ms.Tensor(np.random.random((2, 4)).astype(np.float32))
    z2 = ms.Tensor(np.random.random((2, 4)).astype(np.float32))
    x2_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    y2_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    z2_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)

    net_s = Net1()
    out_s11, out_s12 = net_s(x1, y1, z1)
    out_s21, out_s22 = net_s(x2, y2, z2)

    net_d = Net1()
    net_d.set_inputs(x1_dyn, y1_dyn, z1_dyn)
    out_d11, out_d12 = net_d(x1, y1, z1)
    net_d.set_inputs(x2_dyn, y2_dyn, z2_dyn)
    out_d21, out_d22 = net_d(x2, y2, z2)

    assert np.allclose(out_s11.asnumpy(), out_d11.asnumpy())
    assert np.allclose(out_s12.asnumpy(), out_d12.asnumpy())
    assert np.allclose(out_s21.asnumpy(), out_d21.asnumpy())
    assert np.allclose(out_s22.asnumpy(), out_d22.asnumpy())

    net_d1 = Net1()
    net_d1.set_inputs(y=y1_dyn)
    out_d1_11, out_d1_12 = net_d1(x1, y1, z1)
    net_d1.set_inputs(x=x2_dyn, y=y2_dyn)
    out_d1_21, out_d1_22 = net_d1(x2, y2, z2)

    assert np.allclose(out_s11.asnumpy(), out_d1_11.asnumpy())
    assert np.allclose(out_s12.asnumpy(), out_d1_12.asnumpy())
    assert np.allclose(out_s21.asnumpy(), out_d1_21.asnumpy())
    assert np.allclose(out_s22.asnumpy(), out_d1_22.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dyn_mul_case2(mode):
    """
    Feature: incremental mode support different unset input.
    Description: Test incremental mode support different unset input.
    Expectation: the result match with static result.
    """
    ms.set_context(mode=mode)

    x1 = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    y1 = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    z1 = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    y1_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)

    x2 = ms.Tensor(np.random.random((2, 4)).astype(np.float32))

    net_s = Net1()
    out_s11, out_s12 = net_s(x1, y1, z1)
    out_s21, out_s22 = net_s(x2, y1, z1)

    net_d1 = Net1()
    net_d1.set_inputs(y=y1_dyn)
    out_d1_11, out_d1_12 = net_d1(x1, y1, z1)
    out_d1_21, out_d1_22 = net_d1(x2, y1, z1)

    assert np.allclose(out_s11.asnumpy(), out_d1_11.asnumpy())
    assert np.allclose(out_s12.asnumpy(), out_d1_12.asnumpy())
    assert np.allclose(out_s21.asnumpy(), out_d1_21.asnumpy())
    assert np.allclose(out_s22.asnumpy(), out_d1_22.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_cell_jit_case(mode):
    """
    Feature: incremental mode support different unset input.
    Description: Test incremental mode support different unset input.
    Expectation: the result match with static result.
    """
    ms.set_context(mode=mode)

    class InnerNetStatic(ms.nn.Cell):
        def construct(self, x, y, z):
            add1 = ms.ops.add(x, y)
            return ms.ops.add(add1, z)

    class InnerNetDynamic(ms.nn.Cell):
        @ms.jit(input_signature={"y": ms.Tensor(shape=[None, None, None], dtype=ms.float32)})
        def construct(self, x, y, z):
            add1 = ms.ops.add(x, y)
            return ms.ops.add(add1, z)

    x = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    y = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))
    z = ms.Tensor(np.random.random((2, 3, 4)).astype(np.float32))

    net_s = InnerNetStatic()
    out_s1, out_s2 = net_s(x, y, z)
    net_dyn = InnerNetDynamic()
    out_d1, out_d2 = net_dyn(x, y, z)

    assert np.allclose(out_s1.asnumpy(), out_d1.asnumpy())
    assert np.allclose(out_s2.asnumpy(), out_d2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_single_parameter_case(mode):
    """
    Feature: single parameter case.
    Description: Test single parameter case.
    Expectation: the result match with static result.
    """
    ms.set_context(mode=mode)

    x = ms.Tensor(np.random.random((2, 3)).astype(np.float32))
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)

    class SimpleNet(ms.nn.Cell):
        def construct(self, only_input):
            return ms.ops.add(only_input, 2)

    @ms.jit(input_signature=(x_dyn))
    def simple_func(only_input):
        return ms.ops.add(only_input, 2)

    @ms.jit(input_signature=x_dyn)
    def simple_func1(only_input):
        return ms.ops.add(only_input, 2)

    @ms.jit(input_signature={"only_input": x_dyn})
    def simple_func2(only_input):
        return ms.ops.add(only_input, 2)

    net_s = SimpleNet()
    out_s = net_s(x)

    net_d = SimpleNet()
    net_d.set_inputs(x_dyn)
    out_d = net_d(x)
    assert np.allclose(out_s.asnumpy(), out_d.asnumpy())

    out_func = simple_func(x)
    assert np.allclose(out_s.asnumpy(), out_func.asnumpy())
    out_func1 = simple_func1(x)
    assert np.allclose(out_s.asnumpy(), out_func1.asnumpy())
    out_func2 = simple_func2(x)
    assert np.allclose(out_s.asnumpy(), out_func2.asnumpy())
