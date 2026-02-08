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

"""
dvm test case
"""

import numpy as np
import os
import shutil
import pytest
from mindspore import context
from mindspore import Tensor, nn, JitConfig
from mindspore import Parameter
import mindspore as ms
from mindspore import ops
import mindspore.ops.operations as P
from tests.st.graph_kernel.gk_utils import AssertGKEnable
from tests.mark_utils import arg_mark

ascend_grad_overflow = P.IsFinite()


def tensor_ascend_grad_overflow(grad):
    status = ascend_grad_overflow(grad)
    base = Tensor(1.0, dtype=ms.float32)
    output = base - status.all()
    output = P.Reshape()(output, (1,))
    return output


class ComplexNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.greater = P.Greater()
        self.select = P.Select()
        self.gelu = P.GeLU()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reduce_mean = P.ReduceMean()
        self.addn = P.AddN()

    def construct(self, x, y):
        a = ops.auto_generate.gen_ops_def.add_ext(x, y, 0.1) + 4
        b = x - y - 5
        c = self.gelu(x)
        d = self.reduce_sum(c, (0,))
        e = self.greater(a, b)
        f = self.select(e, a * a, b + 4)
        a_overflow = tensor_ascend_grad_overflow(a)
        b_overflow = tensor_ascend_grad_overflow(b)
        d_overflow = tensor_ascend_grad_overflow(d)
        g = self.addn((a_overflow, b_overflow, a_overflow))
        return f, d, g, d_overflow


def get_output(net, args, args_dyn=None, enable_graph_kernel=False):
    jit_level = "O1" if enable_graph_kernel else "O0"
    context.set_context(jit_config={"jit_level": jit_level})
    with AssertGKEnable(enable_graph_kernel):
        net_obj = net()
        if args_dyn:
            net_obj.set_inputs(*args_dyn)
        output = net_obj(*args)
    return output


def fuse(shape1, shape2, dtype):
    np.random.seed(1)
    i0 = Tensor(np.random.uniform(1, 2, shape1).astype(dtype))
    i1 = Tensor(np.random.uniform(1, 2, shape2).astype(dtype))
    expect = get_output(ComplexNet, [i0, i1], enable_graph_kernel=False)
    expects = [e.asnumpy() for e in expect]
    output = get_output(ComplexNet, [i0, i1], enable_graph_kernel=True)
    outputs = [o.asnumpy() for o in output]
    if dtype == np.float32:
        eps = 1e-4
    elif dtype == np.float16:
        eps = 1e-3
    else:
        eps = 0
    np.testing.assert_allclose(expects[0], outputs[0], eps, eps)
    np.testing.assert_allclose(expects[1], outputs[1], eps, eps)
    np.testing.assert_allclose(expects[2], outputs[2], 0, 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=46.0)
@pytest.mark.parametrize("shape1, shape2", [((32, 1024), (32, 1024)), ((1, 32, 1), (256, 1, 64))])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_easy_fuse_dvm(shape1, shape2, dtype):
    """
    Feature: easy test case for graph_kernel in Ascend.
    Description: ascend test case, use graph_kernel execute ops.
    Expectation: the result match with close graph_kernel result
    """
    context.set_context(mode=context.GRAPH_MODE)
    fuse(shape1, shape2, dtype)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x0, x1, x2):
        y0 = self.mul(x0, x1)
        y1 = self.add(y0, x2)
        return y1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_dvm_dynamic_shape():
    """
    Feature: dynamic shape test case
    Description: test dvm dynamic shape
    Expectation: the result match with expect
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    x0 = np.random.normal(0, 1, (8, 32)).astype(np.float16)
    x1 = np.random.normal(0, 1, (8, 1)).astype(x0.dtype)
    x2 = np.random.normal(0, 1, (1, 32)).astype(x0.dtype)
    args = [Tensor(x0), Tensor(x1), Tensor(x2)]
    args_dyn = [Tensor(shape=(None, None), dtype=ms.float16),
                Tensor(shape=(None, 1), dtype=ms.float16),
                Tensor(shape=(1, None), dtype=ms.float16)]
    expect = get_output(Net, args, args_dyn, enable_graph_kernel=False)
    context.set_context(graph_kernel_flags="--dump_as_text")
    output = get_output(Net, args, args_dyn, enable_graph_kernel=True)
    dump_dir = "./graph_kernel_dump"
    if os.path.isdir(dump_dir):
        shutil.rmtree(dump_dir)
    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 1e-3, 1e-3)


class NetD(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()
        self.add = ops.Add()

    def construct(self, x0, x1):
        y0 = self.reshape(x0, (-1, 1))
        y1 = self.add(y0, x1)
        return y1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_dvm_multiple_run():
    """
    Feature: dynamic shape test case
    Description: test dvm dynamic shape with different input shapes
    Expectation: the result match with expect
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O1"},
                        graph_kernel_flags="--enable_cluster_ops=Reshape")
    x0_dyn = Tensor(shape=(None,), dtype=ms.float16)
    x1_dyn = Tensor(shape=(None,), dtype=ms.float16)
    x0 = np.random.normal(0, 1, (4,)).astype(np.float16)
    x1 = np.random.normal(0, 1, (8,)).astype(x0.dtype)
    x2 = np.random.normal(0, 1, (6,)).astype(np.float16)
    x3 = np.random.normal(0, 1, (2,)).astype(x2.dtype)
    with AssertGKEnable(True):
        net = NetD()
        net.set_inputs(x0_dyn, x1_dyn)
        output1 = net(Tensor(x0), Tensor(x1))
        output1 = output1.asnumpy()
        output2 = net(Tensor(x2), Tensor(x3))
        output2 = output2.asnumpy()
    expect1 = x0.reshape((-1, 1)) + x1
    expect2 = x2.reshape((-1, 1)) + x3
    assert np.allclose(expect1, output1, 1e-3, 1e-3)
    assert np.allclose(expect2, output2, 1e-3, 1e-3)


class NetT(nn.Cell):
    def __init__(self, trans):
        super().__init__()
        self.trans = trans

    def construct(self, x0):
        y0 = ops.Transpose()(x0, self.trans[0])
        y1 = ops.Transpose()(y0, self.trans[1])
        return y1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_dvm_transpose():
    """
    Feature: Transpose test case
    Description: test dvm Transpose optimize
    Expectation: the result match with expect
    """
    np.random.seed(1)
    enable_graph_kernel = True
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O1"},
                        graph_kernel_flags="--enable_cluster_ops=Transpose")
    x0 = np.random.normal(0, 1, (16, 32, 16)).astype(np.float16)
    trans = [(1, 0, 2), (0, 2, 1)]
    with AssertGKEnable(enable_graph_kernel):
        net = NetT(trans)
        net.set_jit_config(JitConfig(jit_level="O1"))
        output = net(Tensor(x0))
        output = output.asnumpy()
    expect = np.transpose(np.transpose(x0, trans[0]), trans[1])
    assert np.allclose(expect, output, 1e-3, 1e-3)


class NetBool(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cond = Tensor(np.array(False))

    def construct(self, x0, x1, x2, x3, x4):
        y0 = ops.Select()(self.cond, x0, x1)
        y1 = ops.BroadcastTo((3, 1, 1, 1))(x2)
        y2 = ops.Select()(y1, x3, x4)
        y3 = ops.Mul()(y0, y2)
        return y3


class SelectNet(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.param = Parameter(Tensor(np.ones(shape), dtype=ms.float16), "param")

    def construct(self, x0, x1, x2, x3):
        y0 = ops.Add()(x0, x1)
        y1 = ops.Select()(x2, x3, y0)
        y2 = ops.Assign()(self.param, y1)
        return y2


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=34.0)
def test_dvm_bool():
    """
    Feature: Boolean type test case
    Description: test dvm boolean data type
    Expectation: the result match with expect
    """

    def case1():
        np.random.seed(1)
        x0 = np.random.normal(0, 1, (3, 1, 1, 1)).astype(np.float16)
        x1 = np.random.normal(0, 1, (3, 1, 1, 1)).astype(np.float16)
        x2 = np.array(True)
        x3 = np.random.normal(0, 1, (3, 1, 1, 1)).astype(np.float16)
        x4 = np.random.normal(0, 1, (3, 1, 1, 1)).astype(np.float16)
        with AssertGKEnable(True):
            net = NetBool()
            output = net(Tensor(x0), Tensor(x1), Tensor(x2), Tensor(x3), Tensor(x4))
            output = output.asnumpy()
        expect = x1 * x3
        assert np.allclose(expect, output, 1e-3, 1e-3)

    def case2():
        x0 = np.random.normal(0, 1, (1, 4, 8192, 96)).astype(np.float16)
        x1 = np.random.normal(0, 1, (1, 4, 8192, 96)).astype(np.float16)
        x2 = np.random.randint(2, size=(1,), dtype=bool)
        x3 = np.random.normal(0, 1, (1, 4, 8192, 96)).astype(np.float16)
        net = SelectNet((1, 4, 8192, 96))
        _ = net(Tensor(x0), Tensor(x1), Tensor(x2), Tensor(x3))
        output = net.param.asnumpy()
        expect = np.select(x2, x3, x0 + x1)
        assert np.allclose(expect, output, 1e-3, 1e-3)

    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O1"})
    context.set_context(graph_kernel_flags="--dump_as_text")
    case1()
    case2()
    dump_dir = "./graph_kernel_dump"
    if os.path.isdir(dump_dir):
        shutil.rmtree(dump_dir)


class NetPow(nn.Cell):
    def __init__(self):
        super().__init__()
        self.const0 = Tensor(2, dtype=ms.float32)
        self.const1 = Tensor(10000, dtype=ms.float32)
        self.const2 = Tensor(1, dtype=ms.float32)

    def construct(self, x0, x1):
        y0 = ops.Mul()(x1, self.const0)
        y1 = ops.RealDiv()(y0, x0)
        y2 = ops.Pow()(self.const1, y1)
        y3 = ops.RealDiv()(self.const2, y2)
        return y3


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_fuse_pow():
    """
    Feature: Pow fuse net
    Description: Pow op first input is large const value
    Expectation: the result match with expect
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    x0 = np.array(1.6243454).astype(np.float32)
    x1 = np.random.normal(0, 1, (288,)).astype(np.float32)
    x0_ms = Tensor(x0)
    x1_ms = Tensor(x1)
    expect = get_output(NetPow, [x0_ms, x1_ms], enable_graph_kernel=False)
    expect = expect.asnumpy()
    output = get_output(NetPow, [x0_ms, x1_ms], enable_graph_kernel=True)
    output = output.asnumpy()
    assert np.allclose(expect, output, 1e-4, 1e-4, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=188.0)
def test_hsigmoid():
    """
    Feature: HSigmoid
    Description: test O1 HSigmoid precision
    Expectation: the result match with expect
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    x0 = np.random.randn(2, 20, 10, 22, 35, 8, 10).astype(np.float32)
    expect = np.maximum(np.minimum(x0 / 6.0 + 0.5, 1.0), 0.0)
    x0_ms = Tensor(x0)
    output = get_output(nn.HSigmoid, [x0_ms], enable_graph_kernel=True)
    output = output.asnumpy()
    assert np.allclose(expect, output, 1e-4, 1e-4, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_fuse_virtual_node():
    """
    Feature: test split pattern FuseVirtualNode
    Description: Transpose + Assign
    Expectation: the result match with expect
    """

    class Net1(nn.Cell):
        def construct(self, x0, x1):
            y0 = ops.transpose(x0, (1, 0))
            return ops.assign(x1, y0)

    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.normal(0, 1, (10, 20)).astype(np.float32)
    y = np.random.normal(0, 1, (20, 10)).astype(np.float32)
    expect = np.transpose(x, (1, 0))
    output = get_output(Net1, [Tensor(x), Parameter(Tensor(y), name="y")], enable_graph_kernel=True)
    output = output.asnumpy()
    assert np.allclose(expect, output, 1e-4, 1e-4, equal_nan=True)
