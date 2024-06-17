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

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor, nn, JitConfig
import mindspore as ms
import mindspore.ops as ops
import mindspore.ops.operations as P
from tests.st.graph_kernel.gk_utils import AssertGKEnable

ascend_grad_overflow = P.IsFinite()


def tensor_ascend_grad_overflow(grad):
    status = ascend_grad_overflow(grad)
    base = Tensor(1.0, dtype=ms.float32)
    output = base - status.all()
    output = P.Reshape()(output, (1,))
    return output


class ComplexNet(nn.Cell):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.greater = P.Greater()
        self.select = P.Select()
        self.gelu = P.GeLU()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reduce_mean = P.ReduceMean()
        self.addn = P.AddN()

    def construct(self, x, y):
        a = ops.extend.add(x, y, 0.1) + 4
        b = x - y - 5
        c = self.gelu(x)
        d = self.reduce_sum(c, (0,))
        e = self.greater(a, b)
        f = self.select(e, a * a, b + 4)
        a_overflow = tensor_ascend_grad_overflow(a)
        b_overflow = tensor_ascend_grad_overflow(b)
        d_overflow = tensor_ascend_grad_overflow(d)
        g = self.addn((a_overflow, b_overflow, d_overflow))
        return f, d, g


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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("shape1, shape2", [((32, 1024), (32, 1024)), ((44, 1, 47, 1), (1, 34, 1, 91))])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_easy_fuse_dvm(shape1, shape2, dtype):
    """
    Feature: easy test case for graph_kernel in Ascend.
    Description: ascend test case, use graph_kernel execute ops.
    Expectation: the result match with close graph_kernel result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    fuse(shape1, shape2, dtype)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x0, x1, x2):
        y0 = self.mul(x0, x1)
        y1 = self.add(y0, x2)
        return y1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dvm_dynamic_shape():
    """
    Feature: dynamic shape test case
    Description: test dvm dynamic shape
    Expectation: the result match with expect
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x0 = np.random.normal(0, 1, (8, 32)).astype(np.float16)
    x1 = np.random.normal(0, 1, (8, 1)).astype(x0.dtype)
    x2 = np.random.normal(0, 1, (1, 32)).astype(x0.dtype)
    args = [Tensor(x0), Tensor(x1), Tensor(x2)]
    args_dyn = [Tensor(shape=(None, None), dtype=ms.float16),
                Tensor(shape=(None, 1), dtype=ms.float16),
                Tensor(shape=(1, None), dtype=ms.float16)]
    expect = get_output(Net, args, args_dyn, enable_graph_kernel=False)
    output = get_output(Net, args, args_dyn, enable_graph_kernel=True)
    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 1e-3, 1e-3)


class NetD(nn.Cell):
    def __init__(self):
        super(NetD, self).__init__()
        self.reshape = ops.Reshape()
        self.add = ops.Add()

    def construct(self, x0, x1):
        y0 = self.reshape(x0, (-1, 1))
        y1 = self.add(y0, x1)
        return y1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dvm_multiple_run():
    """
    Feature: dynamic shape test case
    Description: test dvm dynamic shape with different input shapes
    Expectation: the result match with expect
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
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
        super(NetT, self).__init__()
        self.trans = trans

    def construct(self, x0):
        y0 = ops.Transpose()(x0, self.trans[0])
        y1 = ops.Transpose()(y0, self.trans[1])
        return y1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dvm_transpose():
    """
    Feature: Transpose test case
    Description: test dvm Transpose optimize
    Expectation: the result match with expect
    """
    np.random.seed(1)
    enable_graph_kernel = True
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
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
        super(NetBool, self).__init__()
        self.cond = Tensor(np.array(False))

    def construct(self, x0, x1, x2, x3, x4):
        y0 = ops.Select()(self.cond, x0, x1)
        y1 = ops.BroadcastTo((3, 1, 1, 1))(x2)
        y2 = ops.Select()(y1, x3, x4)
        y3 = ops.Mul()(y0, y2)
        return y3


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dvm_bool():
    """
    Feature: Boolean type test case
    Description: test dvm boolean data type
    Expectation: the result match with expect
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O1"})
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
