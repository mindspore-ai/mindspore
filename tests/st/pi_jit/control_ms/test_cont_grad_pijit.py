import numpy as np
import pytest
from mindspore import dtype as ms
from mindspore import Tensor
from mindspore import context
from mindspore import nn, jit
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P


grad_by_list = C.GradOperation(get_by_list=True)
grad_all = C.GradOperation(get_all=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_const_param_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()

        def construct(self, x, y):
            while x < y:
                z = self.mul(x, x)
                x = self.add(z, 1)
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    idx = Tensor([1.1], dtype=ms.float32)
    end = Tensor([8.0], dtype=ms.float32)
    graph_output = net(idx, end)
    expect_one = np.array([1.14433983e+02], dtype=np.float32)
    expect_two = np.array([0], dtype=np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect_one, 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect_two, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_variable_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()

        def construct(self, x, y):
            while x < y:
                z = self.mul(x, x)
                x = self.add(z, y)
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    idx = Tensor([1.1], dtype=ms.float32)
    end = Tensor([8.0], dtype=ms.float32)
    graph_output = net(idx, end)
    expect_one = np.array([2.20000005e+00], dtype=np.float32)
    expect_two = np.array([1.00000000e+00], dtype=np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect_one, 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect_two, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_forward():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        @jit(mode="PIJit")
        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                part = x[idx, :, :]
                max_num = self.max(part)
                x[idx, :, 0:2] = max_num
                out = out + x + self.param
                idx = idx + 1
            return out

    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MyWhileNet()
    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(2), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    graph_output = net(idx, end, x)
    expect = np.array([[[6, 8], [10, 12]], [[19, 22], [25, 28]]], dtype=np.int32)
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_endless_case():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        @jit(mode="PIJit")
        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                part = x[idx, :, :]
                out = out + part
                idx = idx + 1
            return out

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(2), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net = MyWhileNet()
    graph_output = net(idx, end, x)
    expect = np.array([[[4, 6], [8, 10]],
                       [[4, 6], [8, 10]]]).astype(np.float32)
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                part = x[idx, :, :]
                max_num = self.max(part)
                x[idx, :, 0:2] = max_num
                out = out + x + self.param
                idx = idx + 1
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(2), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    graph_output = net(idx, end, x)
    expect = np.array([[[2, 2], [2, 2]], [[2, 2], [2, 2]]], dtype=np.int32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_forward_with_const_branch():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.reduce = P.ReduceSum()

        @jit(mode="PIJit")
        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                if 2 > 1:
                    out = out + self.param
                else:
                    out = out + idx + self.param
                idx = idx + 1
            return out

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(4), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = while_net
    graph_output = net(idx, end, x)

    expect = np.array([[[0, 4], [8, 12]],
                       [[16, 20], [24, 28]]]).astype(np.float32)
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_opt_endless():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """

    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.reduce = P.ReduceSum()
            self.addn = P.AddN()

        def construct(self, idx, end, x):
            addn1 = self.addn((x, x, x))
            out = addn1
            while idx < end:
                out = self.addn((out, addn1))
                idx = idx + 1
            out = self.addn((out, x))
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(4), dtype=ms.int32)
    x = Tensor(np.ones([2, 2, 2]).astype(np.float32) * 3, dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)

    expect1 = 0
    expect2 = 0
    expect3 = np.array([[[16, 16], [16, 16]],
                        [[16, 16], [16, 16]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect1, 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect2, 0.0001, 0.0001)
    assert np.allclose(graph_output[2].asnumpy(), expect3, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_no_while_call():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.reduce = P.ReduceSum()

        @jit(mode="PIJit")
        def construct(self, idx, end, x):
            out = self.zero
            if 2 > 1:
                out = out + self.param
            else:
                out = out + idx + self.param
            return out

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(4), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = while_net
    graph_output = net(idx, end, x)

    expect = np.array([[[0, 1], [2, 3]],
                       [[4, 5], [6, 7]]]).astype(np.float32)
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_grad_with_const_branch():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.reduce = P.ReduceSum()

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                if 2 > 1:
                    out = out + self.param
                else:
                    out = out + idx + self.param
                idx = idx + 1
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(4), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)

    expect = np.array([[[4, 4], [4, 4]],
                       [[4, 4], [4, 4]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_for_while_with_param_grad_with_const_branch():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.reduce = P.ReduceSum()
            self.start = Tensor(np.array(0), dtype=ms.int32)

        def construct(self, idx, end, x):
            out = self.zero
            for _ in range(0, 2):
                idx = self.start
                while idx < end:
                    if 2 > 1:
                        out = out + self.param
                    else:
                        out = out + idx + self.param
                    idx = idx + 1
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(4), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)

    expect = np.array([[[8, 8], [8, 8]],
                       [[8, 8], [8, 8]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_for_while_with_param_grad_basic():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.reduce = P.ReduceSum()
            self.start = Tensor(np.array(0), dtype=ms.int32)

        def construct(self, idx, end, x):
            out = self.zero
            for _ in range(0, 2):
                idx = self.start
                while idx < end:
                    out = out + self.param
                    idx = idx + 1
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(4), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)
    expect = np.array([[[8, 8], [8, 8]],
                       [[8, 8], [8, 8]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_for_while_with_param_grad_normal():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.reduce = P.ReduceSum()
            self.start = Tensor(np.array(0), dtype=ms.int32)

        def construct(self, idx, end, x):
            out = x
            for _ in range(0, 2):
                idx = self.start
                while idx < end:
                    out = out + self.param
                    idx = idx + 1
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(4), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)
    expect = np.array([[[8, 8], [8, 8]],
                       [[8, 8], [8, 8]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_basic_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.t2 = Tensor(np.array(2), dtype=ms.float32)

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                out = out + self.param
                idx = idx + 1
            return out + self.param

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(3), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)
    expect = np.array([[[4, 4], [4, 4]],
                       [[4, 4], [4, 4]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_basic_grad_mul():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.ones(([2, 2, 2])), ms.float32)
            self.t2 = Tensor(np.array(2), dtype=ms.float32)

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                out = out * self.param
                idx = idx + 1
            return out + self.param

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(3), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)
    expect = np.array([[[1, 4], [13, 28]],
                       [[49, 76], [109, 148]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_basic_grad_two():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.weight = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="loss")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.t2 = Tensor(np.array(2), dtype=ms.float32)

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                out = out + self.param + self.weight
                idx = idx + 1
            return out + self.param

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(3), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)

    expect1 = np.array([[[4, 4], [4, 4]],
                        [[4, 4], [4, 4]]]).astype(np.float32)
    expect2 = np.array([[[3, 3], [3, 3]],
                        [[3, 3], [3, 3]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect1, 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect2, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_basic_grad_three():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.weight = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="loss")
            self.key = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="key")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)
            self.t2 = Tensor(np.array(2), dtype=ms.float32)

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                out = out + self.param + self.weight + self.key
                idx = idx + 1
            return out + self.param

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(3), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)
    expect1 = np.array([[[4, 4], [4, 4]],
                        [[4, 4], [4, 4]]]).astype(np.float32)
    expect2 = np.array([[[3, 3], [3, 3]],
                        [[3, 3], [3, 3]]]).astype(np.float32)
    expect3 = np.array([[[3, 3], [3, 3]],
                        [[3, 3], [3, 3]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect1, 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect2, 0.0001, 0.0001)
    assert np.allclose(graph_output[2].asnumpy(), expect3, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_with_param_grad_not_enter_while():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(2, ms.float32), name="weight")
            self.zero = Tensor(0, ms.float32)

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                out = out + self.param * 3
                idx = idx + 1
            return out + self.param

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    idx = Tensor(np.array(3), dtype=ms.int32)
    end = Tensor(np.array(0), dtype=ms.int32)
    x = Tensor(2, dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)

    assert np.allclose(graph_output[0].asnumpy(), 1, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_param_if_by_if_grad_inputs():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        def construct(self, a, b, x):
            out = self.zero
            if a < b:
                out = out + x + self.param * 4
            if a == b:
                out = out + x * 3 + self.param * 3
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(0), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = GradNet(if_net)
    graph_output = net(idx, end, x)
    expect1 = Tensor(np.array(0), dtype=ms.int32)
    expect2 = Tensor(np.array(0), dtype=ms.int32)
    expect3 = np.array([[[3, 3], [3, 3]],
                        [[3, 3], [3, 3]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect1.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect2.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(graph_output[2].asnumpy(), expect3, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_param_if_by_if_grad_parameter():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        def construct(self, a, b, x):
            out = self.zero
            if a < b:
                out = out + x + self.param * 2
            if a == b:
                out = out + x * 3 + self.param
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_by_list(self.net, self.weights)(*inputs)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(2), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = GradNet(if_net)
    graph_output = net(idx, end, x)

    expect = np.array([[[2, 2], [2, 2]],
                       [[2, 2], [2, 2]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_param_if_by_if_grad_param_excute_null():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        def construct(self, a, b, x):
            out = self.zero
            if a < b:
                out = out + x + self.param * 2
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_by_list(self.net, self.weights)(*inputs)

    idx = Tensor(np.array(4), dtype=ms.int32)
    end = Tensor(np.array(0), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = GradNet(if_net)
    graph_output = net(idx, end, x)

    expect = np.array([[[0, 0], [0, 0]],
                       [[0, 0], [0, 0]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_if_return_inside_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        def construct(self, a, b, x):
            out = self.zero
            if a < b:
                return out + x + self.param
            if a == b:
                return out + self.param * 2
            return out + self.param * 3

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, *inputs):
            return grad_by_list(self.net, self.weights)(*inputs)

    idx = Tensor(np.array(1), dtype=ms.int32)
    end = Tensor(np.array(0), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(np.float32), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = GradNet(if_net)
    graph_output = net(idx, end, x)

    expect = np.array([[[3, 3], [3, 3]],
                       [[3, 3], [3, 3]]]).astype(np.float32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_if_forward():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.div = P.RealDiv()

        #TODO enable PIJit
        def construct(self, a, b, x):
            if a < b:
                a = self.add(a, b)
            else:
                a = self.sub(a, b)
            if a == x:
                a = self.mul(a, b)
            else:
                a = self.div(a, b)
            if b == x:
                b = self.add(a, b)
            else:
                b = self.add(a, x)
            a = a * b
            out = a + b + x
            return out

    idx = Tensor(np.array(2), dtype=ms.float32)
    end = Tensor(np.array(3), dtype=ms.float32)
    x = Tensor(np.array(4), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = if_net
    graph_output = net(idx, end, x)
    expect = 19.11111
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_if_forward_use_namespace():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.div = P.RealDiv()

        #TODO: enable PIJit
        def construct(self, a, b, x):
            if a < b:
                a = P.Add()(a, b)
            else:
                a = P.Sub()(a, b)
            if a == x:
                a = P.Mul()(a, b)
            else:
                a = P.RealDiv()(a, b)
            if b == x:
                b = P.Add()(a, b)
            else:
                b = P.Add()(a, x)
            a = a * b
            out = a + b + x
            return out

    idx = Tensor(np.array(2), dtype=ms.float32)
    end = Tensor(np.array(3), dtype=ms.float32)
    x = Tensor(np.array(0), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = if_net
    graph_output = net(idx, end, x)
    expect = 4.444444
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_if_forward_use_global_op():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.div = P.RealDiv()

        #TODO: enable PIJit
        def construct(self, a, b, x):
            add = P.Add()
            sub = P.Sub()
            mul = P.Mul()
            div = P.RealDiv()
            if a < b:
                a = add(a, b)
            else:
                a = sub(a, b)
            if a == x:
                a = mul(a, b)
            else:
                a = div(a, b)
            if b == x:
                b = add(a, b)
            else:
                b = add(a, x)
            a = a * b
            out = a + b + x
            return out

    idx = Tensor(np.array(2), dtype=ms.float32)
    end = Tensor(np.array(3), dtype=ms.float32)
    x = Tensor(np.array(0), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = if_net
    graph_output = net(idx, end, x)

    expect = 4.444444
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_for_with_if_by_if_forward():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()

        @jit(mode="PIJit")
        def construct(self, a, b, x):
            for _ in range(0, 4):
                if a < b:
                    a = self.add(a, b)
                else:
                    b = self.sub(b, x)
            a = a * b
            out = a + b + x
            return out

    idx = Tensor(np.array(2), dtype=ms.float32)
    end = Tensor(np.array(3), dtype=ms.float32)
    x = Tensor(np.array(0), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = if_net
    graph_output = net(idx, end, x)

    expect = 18.0
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_for_with_if_by_if_forward_namespace():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.div = P.RealDiv()

        @jit(mode="PIJit")
        def construct(self, a, b, x):
            for _ in range(0, 6):
                if a < b:
                    a = P.Add()(a, b)
                else:
                    b = P.Sub()(b, x)
            a = a * b
            out = a + b + x
            return out

    idx = Tensor(np.array(2), dtype=ms.float32)
    end = Tensor(np.array(3), dtype=ms.float32)
    x = Tensor(np.array(0), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = if_net
    graph_output = net(idx, end, x)

    expect = 18.0
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_if_forward_const_branch_inner():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.div = P.RealDiv()

        #TODO: enable PIJit
        def construct(self, a, b, x):
            add = P.Add()
            sub = P.Sub()
            mul = P.Mul()
            div = P.RealDiv()
            if a < b:
                a = add(a, b)
            else:
                a = sub(a, b)
            if 2 > 1:
                a = mul(a, b)
            else:
                a = div(a, b)
            if b == x:
                b = add(a, b)
            else:
                b = add(a, x)
            a = a * b
            out = a + b + x
            return out

    idx = Tensor(np.array(2), dtype=ms.float32)
    end = Tensor(np.array(3), dtype=ms.float32)
    x = Tensor(np.array(0), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = if_net
    graph_output = net(idx, end, x)

    expect = 240.0
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_if_forward_all_const_branch():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyIfByIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.div = P.RealDiv()

        @jit(mode="PIJit")
        def construct(self, a, b, x):
            add = P.Add()
            sub = P.Sub()
            mul = P.Mul()
            div = P.RealDiv()
            if 2 < 12:
                a = add(a, b)
            else:
                a = sub(a, b)
            if 2 > 1:
                a = mul(a, b)
            else:
                a = div(a, b)
            if 2 == 1:
                b = add(a, b)
            else:
                b = add(a, x)
            a = a * b
            out = a + b + x
            return out

    idx = Tensor(np.array(2), dtype=ms.float32)
    end = Tensor(np.array(3), dtype=ms.float32)
    x = Tensor(np.array(0), dtype=ms.float32)
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = MyIfByIfNet()
    net = if_net
    graph_output = net(idx, end, x)

    expect = 240.0
    assert np.allclose(graph_output.asnumpy(), expect, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_const_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, *inputs):
            a = 1
            b = 2
            if a > 0:
                b = 1
            a += b
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.PYNATIVE_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_if_const_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, *inputs):
            a = 1
            b = 2
            if a > 0:
                b = 1
            if a < 0:
                b = 0
            if a == 0:
                b = 3
            a += b
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.PYNATIVE_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_while_const_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, *inputs):
            a = 1
            while a > 1:
                a = a - 1
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.PYNATIVE_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_if_by_while_const_grad():
    """
    Feature: Get container with grad from abstract.
    Description: Watching the container with grad func graph from abstract.
    Expectation: Output correct.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        @jit(mode="PIJit")
        def construct(self, *inputs):
            a = 1
            b = 2
            if a > 0:
                b = 0
            while a > 1:
                a = a - 1
            a += b
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.PYNATIVE_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)
