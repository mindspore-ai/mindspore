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
"""Test morph with custom bprop"""

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Parameter, ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_morph_custom_bprop_001():
    """
    Feature: Morph.
    Description: Custom bprop of Morph matches forward function; gradients and output match baseline.
    Expectation: Forward and backward results are numerically consistent with equivalent native implementation.
    """
    def infer_dtype(*args):
        return args[0]

    def infer_shape(*args):
        return args[0]

    def fn(x, y):
        return x * y

    def bprop_fn(x, y, out, dout):
        return (dout * y, dout * x)

    class TestNet0Morph(nn.Cell):
        def __init__(self, bprop_fn=None):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")
            self.morph = ops.Morph(
                fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)

        def construct(self, x):
            y = x * self.weight0
            z = self.morph(y, x)
            out = z * self.weight1
            return out

    x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(x)
    net = TestNet0Morph(bprop_fn)
    out_forward = ms.jit(net)(input_x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    bwd_out = ms.jit(grad_net)(input_x)

    class TestNet0(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")

        def construct(self, x):
            y = x * self.weight0
            z = y * x
            out = z * self.weight1
            return out

    net_1 = TestNet0()
    out_forward_1 = ms.jit(net_1)(input_x)
    grad_net_1 = grad_op(net_1, net_1.trainable_params())
    bwd_out_1 = ms.jit(grad_net_1)(input_x)

    assert np.allclose(out_forward.asnumpy(),
                       out_forward_1.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[0][0].asnumpy(),
                       bwd_out_1[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][0].asnumpy(),
                       bwd_out_1[1][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][1].asnumpy(),
                       bwd_out_1[1][1].asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_morph_custom_bprop_multi_outputs():
    """
    Feature: Morph.
    Description: Morph with multi-output forward and custom bprop.
    Expectation: Forward and backward results match native implementation within tolerance.
    """
    def infer_dtype(*args):
        return args[0], args[0]

    def infer_shape(*args):
        return args[0], args[0]

    def fn(x, y):
        return x * y, x * y

    def bprop_fn(x, y, out, dout):
        return (2 * dout[0] * y, 2 * dout[1] * x)

    class TestNet0Morph(nn.Cell):
        def __init__(self, bprop_fn=None):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")
            self.morph = ops.Morph(
                fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)

        def construct(self, x):
            y = x * self.weight0
            z, w = self.morph(y, x)
            out = z * w
            return out

    x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(x, ms.float32)
    net = TestNet0Morph(bprop_fn)
    out_forward = ms.jit(net)(input_x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    bwd_out = ms.jit(grad_net)(input_x)

    class TestNet0(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")

        def construct(self, x):
            y = x * self.weight0
            z, w = x * y, x * y
            out = z * w
            return out

    net_1 = TestNet0()
    out_forward_1 = ms.jit(net_1)(input_x)
    grad_net_1 = grad_op(net_1, net_1.trainable_params())
    bwd_out_1 = ms.jit(grad_net_1)(input_x)

    assert np.allclose(out_forward.asnumpy(),
                       out_forward_1.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[0][0].asnumpy(),
                       bwd_out_1[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][0].asnumpy(),
                       bwd_out_1[1][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][1].asnumpy(),
                       bwd_out_1[1][1].asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_morph_with_ctrl_flow_in_custom_bprop():
    """
    Feature: Morph.
    Description: Custom bprop contains control flow (if-else).
    Expectation: Forward and backward results match native implementation.
    """
    def infer_dtype(*args):
        return args[0]

    def infer_shape(*args):
        return args[0]

    def fn(x, y):
        return x * y

    def bprop_fn(x, y, out, dout):
        if ops.ReduceSum()(y) == ops.ReduceSum()(x):
            return y * dout * 2, x * dout * 2
        return y * dout, x * dout

    class TestNet0Morph(nn.Cell):
        def __init__(self, bprop_fn=None):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")
            self.morph = ops.Morph(
                fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)

        def construct(self, x):
            y = x * self.weight0
            z = self.morph(y, x)
            out = z * y
            return out

    x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(x, ms.float32)
    net = TestNet0Morph(bprop_fn)
    out_forward = ms.jit(net)(input_x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    bwd_out = ms.jit(grad_net)(input_x)

    class TestNet0(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")

        def construct(self, x):
            y = x * self.weight0
            z = x * y
            out = z * y
            return out

    net_1 = TestNet0()
    out_forward_1 = ms.jit(net_1)(input_x)
    grad_net_1 = grad_op(net_1, net_1.trainable_params())
    bwd_out_1 = ms.jit(grad_net_1)(input_x)

    assert np.allclose(out_forward.asnumpy(),
                       out_forward_1.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[0][0].asnumpy(),
                       bwd_out_1[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][0].asnumpy(),
                       bwd_out_1[1][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][1].asnumpy(),
                       bwd_out_1[1][1].asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_morph_with_side_effect_in_custom_bprop():
    """
    Feature: Morph.
    Description: Custom bprop includes side effect (e.g., print).
    Expectation: Execution succeeds and results match native implementation.
    """
    def infer_dtype(*args):
        return args[0]

    def infer_shape(*args):
        return args[0]

    def fn(x, y):
        return x * y

    def bprop_fn(x, y, out, dout):
        print(out)
        return y * dout, x * dout

    class TestNet0Morph(nn.Cell):
        def __init__(self, bprop_fn=None):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")
            self.morph = ops.Morph(
                fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)

        def construct(self, x):
            y = x * self.weight0
            z = self.morph(y, x)
            out = z * y
            return out

    x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(x, ms.float32)
    net = TestNet0Morph(bprop_fn)
    out_forward = ms.jit(net)(input_x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    bwd_out = ms.jit(grad_net)(input_x)

    class TestNet0(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")

        def construct(self, x):
            y = x * self.weight0
            z = x * y
            out = z * y
            return out

    net_1 = TestNet0()
    out_forward_1 = ms.jit(net_1)(input_x)
    grad_net_1 = grad_op(net_1, net_1.trainable_params())
    bwd_out_1 = ms.jit(grad_net_1)(input_x)

    assert np.allclose(out_forward.asnumpy(),
                       out_forward_1.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[0][0].asnumpy(),
                       bwd_out_1[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][0].asnumpy(),
                       bwd_out_1[1][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][1].asnumpy(),
                       bwd_out_1[1][1].asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_morph_with_custom_fn_and_bprop_fn_not_match():
    """
    Feature: Morph.
    Description: Bprop returns fewer gradients than forward inputs.
    Expectation: A RuntimeError is raised.
    """
    def infer_dtype(*args):
        return args[0]

    def infer_shape(*args):
        return args[0]

    def fn(x, y):
        return x * y

    def bprop_fn(x, y, out, dout):
        return (y * dout,)

    class TestNet0Morph(nn.Cell):
        def __init__(self, bprop_fn=None):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")
            self.morph = ops.Morph(
                fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)

        def construct(self, x):
            y = x * self.weight0
            z = self.morph(y, x)
            out = z * y
            return out

    x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(x, ms.float32)
    net = TestNet0Morph(bprop_fn)
    _ = ms.jit(net)(input_x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    with pytest.raises(RuntimeError):
        _ = ms.jit(grad_net)(input_x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_morph_custom_bprop_fn_inputs_and_outputs_not_match():
    """
    Feature: Morph.
    Description: Bprop signature mismatches forward (missing input argument).
    Expectation: A TypeError is raised.
    """
    def infer_dtype(*args):
        return args[0]

    def infer_shape(*args):
        return args[0]

    def fn(x, y):
        return x * y

    def bprop_fn(x, out, dout):
        return (x * dout, dout)

    class TestNet0Morph(nn.Cell):
        def __init__(self, bprop_fn=None):
            super().__init__()
            self.weight0 = Parameter(
                Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(
                Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")
            self.morph = ops.Morph(
                fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)

        def construct(self, x):
            y = x * self.weight0
            z = self.morph(y, x)
            out = z * y
            return out

    x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(x, ms.float32)
    net = TestNet0Morph(bprop_fn)
    _ = ms.jit(net)(input_x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    with pytest.raises(TypeError) as e:
        _ = ms.jit(grad_net)(input_x)
    assert "The params of function 'bprop' of Primitive or Cell requires the forward "\
        "inputs as well as the 'out' and 'dout'" in str(e.value)
