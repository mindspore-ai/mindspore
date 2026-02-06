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
# ==============================================================================
"""Test morph primitive."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import context, nn, ops, Tensor, Parameter
from tests.mark_utils import arg_mark

np_weight0 = np.array([1.0, 2.0, 3.0])
np_weight1 = np.array([4.0, 5.0, 6.0])
np_input_x = np.array([7.0, 8.0, 9.0])


def infer_dtype(*args):
    return args[0]


def infer_shape(*args):
    return args[0]


default_b = Tensor(2.0, ms.float32)
default_c = Tensor(3.0, ms.float32)
default_d = Tensor(4.0, ms.float32)


def mul_by(*args):
    def inner(a, b=default_b, c=default_c, d=default_d):
        x = args[0] * a
        x = x * b
        x = x * c
        x = x * d
        return x
    return inner


NUMBER_5 = 5
NUMBER_2 = 2


def fn(a, b, c, d):
    return NUMBER_5 * a * b * c * d


def bprop(a, b, c, d, out, dout):
    return (dout * b * c * d * NUMBER_5 * NUMBER_2, dout, dout, dout)


def bprop_call_fn(a, b, c, d, out, dout):
    fn_out = fn(a, b, c, d)
    return (dout * b * c * d * NUMBER_5 * NUMBER_2, dout, dout * fn_out, dout * fn_out)


class TestNet0(nn.Cell):
    def __init__(self, bprop_fn=None):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")
        self.morph = ops.Morph(mul_by(NUMBER_5), infer_shape,
                               infer_dtype, bprop_fn=bprop_fn)

    def construct(self, x):
        o = x * self.weight0
        o = self.morph(o, default_b, default_c, default_d)
        out = o * self.weight1
        return out


class TestNet1(nn.Cell):
    def __init__(self, bprop_fn=None):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")
        self.morph = ops.Morph(mul_by(NUMBER_5), infer_shape,
                               infer_dtype, bprop_fn=bprop_fn)

    def construct(self, x):
        o = x * self.weight0
        o = self.morph(o, d=default_d, b=default_b)
        out = o * self.weight1
        return out


class TestNet2(nn.Cell):
    def __init__(self, bprop_fn=None):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")
        self.morph = ops.Morph(mul_by(NUMBER_5), infer_shape,
                               infer_dtype, bprop_fn=bprop_fn)

    def construct(self, x):
        o = x * self.weight0
        o = self.morph(o, d=default_d, b=default_b)
        o = self.morph(o, c=default_c)
        o = self.morph(o)
        out = o * self.weight1
        return out


class TestNet3(nn.Cell):
    def __init__(self, bprop_fn=None):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")
        self.morph = ops.Morph(fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)

    def construct(self, x):
        o = x * self.weight0
        o = self.morph(o, default_b, default_c, default_d)
        o = self.morph(o, default_b, default_c, default_d)
        o = self.morph(o, default_b, default_c, default_d)
        out = o * self.weight1
        return out


class TestNet4(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight0 = Parameter(
            Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(
            Tensor(np_weight1, ms.float32), name="weight1")
        self.number_5 = NUMBER_5
        self.number_2 = NUMBER_2
        self.morph = ops.Morph(self.morph_fn, infer_shape,
                               infer_dtype, bprop_fn=self.morph_bprop_fn)

    def morph_fn(self, a, b, c, d):
        return self.number_5 * a * b * c * d

    def morph_bprop_fn(self, a, b, c, d, out, dout):
        return (dout * b * c * d * self.number_5 * self.number_2, dout, dout, dout)

    def construct(self, x):
        o = x * self.weight0
        o = self.morph(o, default_b, default_c, default_d)
        out = o * self.weight1
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("net, with_bprop_fn, morph_call_time", [
    (TestNet0(), False, 1),
    (TestNet1(), False, 1),
    (TestNet2(), False, 3),
    (TestNet3(bprop_fn=bprop), True, 3),
    (TestNet3(bprop_fn=bprop_call_fn), True, 3),
    (TestNet4(), True, 1)])
def test_morph_graph_mode(net, with_bprop_fn, morph_call_time):
    """
    Feature: Morph Primitive
    Description: Test morph primitive for graph mode.
    Expectation: Run successfully.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = Tensor(np_input_x, ms.float32)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    bwd_out = grad_net(input_x)
    x_grad = bwd_out[0][0].asnumpy()
    weight0_grad = bwd_out[1][0].asnumpy()
    weight1_grad = bwd_out[1][1].asnumpy()

    grad_factor = NUMBER_2 if with_bprop_fn else 1
    morph_const = NUMBER_5 * default_b * default_c * default_d
    expect_x_grad = np_weight1 * np_weight0 * \
        (morph_const * grad_factor) ** morph_call_time
    expect_weight0_grad = np_weight1 * np_input_x * \
        (morph_const * grad_factor) ** morph_call_time
    expect_weight1_grad = np_input_x * \
        np_weight0 * (morph_const) ** morph_call_time

    assert np.allclose(x_grad, expect_x_grad)
    assert np.allclose(weight0_grad, expect_weight0_grad)
    assert np.allclose(weight1_grad, expect_weight1_grad)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_morph_unsupported_cases():
    """
    Feature: Morph Primitive
    Description: Test morph primitive with unsupported cases.
    Expectation: Correct exception is thrown in unsupported cases.
    """

    def fn1(x, y):
        pass

    def bprop_fn1(x, y, out, dout):
        pass

    _ = ops.Morph(fn1, infer_shape, infer_dtype, bprop_fn1)

    def fn2(x, y=1):
        pass

    def bprop_fn2(x, y):
        pass

    def fn3(x, *args):
        pass

    def bprop_fn3(x, *args, out, dout):
        pass

    def fn4(x, **kwargs):
        pass

    # Can not pass **kwargs to bprop as out/dout can not be set after **kwargs.
    def bprop_fn4(x, out, dout):
        pass

    def fn5(x, *args, y, **kwargs):
        pass

    # Can not pass **kwargs to bprop as out/dout can not be set after **kwargs.
    def bprop_fn5(x, *args, y, out, dout):
        pass

    with pytest.raises(ValueError) as e:
        _ = ops.Morph(fn2, infer_shape, infer_dtype, bprop_fn2)
    assert "Morph `fn` only support positional or keyword parameters with default value is empty" in str(
        e.value)

    with pytest.raises(ValueError) as e:
        _ = ops.Morph(fn3, infer_shape, infer_dtype, bprop_fn3)
    assert "Morph `fn` only support positional or keyword parameters with default value is empty" in str(
        e.value)

    with pytest.raises(ValueError) as e:
        _ = ops.Morph(fn4, infer_shape, infer_dtype, bprop_fn4)
    assert "Morph `fn` only support positional or keyword parameters with default value is empty" in str(
        e.value)

    with pytest.raises(ValueError) as e:
        _ = ops.Morph(fn5, infer_shape, infer_dtype, bprop_fn5)
    assert "Morph `fn` only support positional or keyword parameters with default value is empty" in str(
        e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_morph_pynative_mode():
    """
    Feature: Morph Primitive
    Description: Test morph primitive for pynative mode.
    Expectation: Run successfully.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np_input_x, ms.float32)
    net = TestNet0()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    with pytest.raises(RuntimeError) as e:
        grad_net(input_x)
    assert "Morph is only supported in GRAPH_MODE." in str(e.value)


def morph_fn1(a, b, c, d):
    def morph_fn_inner(a, *, b=None, c=None, **kwargs):
        return (a + b) * (c + kwargs["d"])
    return morph_fn_inner(a, c=c, d=d, b=b)


def morph_fn2(a, b, c, d):
    def morph_fn_inner(a, *args, c=None, **kwargs):
        return (a + args[0]) * (c + kwargs["d"])
    return morph_fn_inner(a, b, d=d, c=c)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("morph_fn", [morph_fn1, morph_fn2])
def test_morph_inner_func_with_complex_args(morph_fn):
    """
    Feature: Morph Primitive
    Description: Verify that Morph can correctly execute an inner function using complex argument signatures,
                 including keyword-only arguments, *args, and **kwargs.
    Expectation: The result should be (a + b) * (c + d) = (2.0 + 3.0) * (4.0 + 5.0) = 45.0.
    """

    class MorphNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.fa = ops.Morph(morph_fn, infer_shape, infer_dtype)

        def construct(self, a, b, c, d):
            return self.fa(a, b, c, d)

    a = Tensor(2.0, dtype=ms.float32)
    b = Tensor(3.0, dtype=ms.float32)
    c = Tensor(4.0, dtype=ms.float32)
    d = Tensor(5.0, dtype=ms.float32)

    net = MorphNet()
    out = ms.jit(net)(a, b, c, d)
    assert np.allclose(out.asnumpy(), 45.0)
