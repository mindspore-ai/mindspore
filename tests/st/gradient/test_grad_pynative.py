# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test function grad in pynative mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore import jit
from mindspore.ops import composite as C
from mindspore.ops import grad, value_and_grad, vmap, get_grad
from mindspore.common import dtype as mstype
from mindspore import Parameter, ParameterTuple

context.set_context(mode=context.PYNATIVE_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x ** 3


class SingleInputMultipleOutputsNet(nn.Cell):
    def construct(self, x):
        return x ** 3, 2 * x


class MultipleInputsSingleOutputNet(nn.Cell):
    def construct(self, x, y, z):
        return x * y * z


class MultipleInputsMultipleOutputsNet(nn.Cell):
    def construct(self, x, y, z):
        return x ** 2 + y ** 2 + z ** 2, x * y * z


class ParamNet(nn.Cell):
    def __init__(self):
        super(ParamNet, self).__init__()
        self.w = Parameter(Tensor([2., 2.]), name="w")
        self.z = Parameter(Tensor([3., 3.]), name="z")

    def construct(self, x):
        res = x * self.w * self.z
        return res


def function(x, y, z):
    return x ** 2 + y ** 2 + z ** 2, x * y * z


def iteration_grad_function(x, y, z):
    return x ** 2 * y * z


@jit
def grad_wrap_with_msfunction(x, y, z):
    output = grad(function)(x, y, z)
    return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_single_output_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with single input and single output net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    real_grad = grad(net)(x)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_multiple_outputs_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with single input and multiple outputs net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputsNet()
    expect_grad = Tensor(np.array([[5, 14], [29, 50]]).astype(np.float32))
    real_grad = grad(net)(x)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_single_output_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with multiple inputs and single output net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    expect_grad1 = Tensor(np.array([[0, 6], [15, -4]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-2, 6], [-3, 8]]).astype(np.float32))
    real_grad = grad(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_multiple_outputs_cell_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with multiple inputs and multiple outputs net in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsMultipleOutputsNet()
    expect_grad1 = Tensor(np.array([[-4, 12], [13, 0]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-2, 12], [7, 6]]).astype(np.float32))
    real_grad = grad(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_iteration_function_pynative():
    """
    Features: Function grad.
    Description: Test calling F.grad iterative with function in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad1 = Tensor(np.array([[0, 12], [30, -8]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-4, 12], [-6, 16]]).astype(np.float32))
    real_grad = grad(grad(iteration_grad_function), grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_wrap_with_msfunction_pynative():
    """
    Features: Function grad.
    Description: Test F.grad wrapped with @jit decorated function in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad = Tensor(np.array([[2, 13], [1, 6]]).astype(np.float32))
    real_grad = grad_wrap_with_msfunction(x, y, z)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_vmap_pynative():
    """
    Features: Function grad.
    Description: Test F.grad vmap  in pynative mode.
    Expectation: No exception.
    """

    def fn(x):
        return x * x

    class VmapNet(nn.Cell):
        def __init__(self, net):
            super(VmapNet, self).__init__()
            self.grad_net = grad(net)

        def construct(self, x):
            res = vmap(self.grad_net, 0, 0)(x)
            return res

    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    ms_net = VmapNet(fn)
    outputs = ms_net(x)
    expect_value = np.array([[2, 4], [6, 8]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_with_grad_position_twice_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with function setting grad_position twice in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    out1 = grad(net, grad_position=0)(x, y, z)
    out2 = grad(net, grad_position=(0, 1))(x, y, z)
    assert isinstance(out1, Tensor)
    assert isinstance(out2, tuple)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_with_weights_twice_pynative():
    """
    Features: GradOperation and grad.
    Description: Test F.grad with different weights twice in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNet()
    grad_fn = C.GradOperation(get_by_list=True)
    weights1 = ParameterTuple(net.trainable_params()[:1])
    weights2 = ParameterTuple(net.trainable_params()[1:])
    expect1 = np.array([3, 6]).astype(np.float32)
    expect2 = np.array([2, 4]).astype(np.float32)
    out1 = grad_fn(net, weights1)(x)
    out2 = grad_fn(net, weights2)(x)
    assert np.allclose(out1[0].asnumpy(), expect1)
    assert np.allclose(out2[0].asnumpy(), expect2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_with_weights_has_aux_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with different weights and has_aux in pynative mode.
    Expectation: No exception.
    """

    class ParamNetAux(nn.Cell):
        def __init__(self):
            super(ParamNetAux, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            res = x * self.w * self.z
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetAux()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    expect_aux1 = np.array([1, 2]).astype(np.float32)
    expect_aux2 = np.array([2, 2]).astype(np.float32)
    res, aux = grad(net, 0, weights, True)(x)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[1][1].asnumpy(), expect_grad_weight2)
    assert np.allclose(aux[0].asnumpy(), expect_aux1)
    assert np.allclose(aux[1].asnumpy(), expect_aux2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_if_with_weights_has_aux_pynative():
    """
    Features: Function grad.
    Description: Test F.grad with different weights and has_aux as well as if case in pynative mode.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            if x[0] == 1:
                res = x * self.w * self.z
            else:
                res = x * x
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = Net()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    expect_aux1 = np.array([1, 2]).astype(np.float32)
    expect_aux2 = np.array([2, 2]).astype(np.float32)
    res, aux = grad(net, 0, weights, True)(x)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[1][1].asnumpy(), expect_grad_weight2)
    assert np.allclose(aux[0].asnumpy(), expect_aux1)
    assert np.allclose(aux[1].asnumpy(), expect_aux2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_nest_with_weights_has_aux_pynative():
    """
    Features: Function value_and_grad.
    Description: Test F.grad with different weights and has_aux as well as nested nets in pynative mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([18, 18]).astype(np.float32)
    expect_grad_weight1 = np.array([9, 18]).astype(np.float32)
    expect_grad_weight2 = np.array([6, 12]).astype(np.float32)
    expect_aux = np.array([6, 12]).astype(np.float32)
    res, aux = grad(net, 0, weights, True)(x)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[1][1].asnumpy(), expect_grad_weight2)
    assert np.allclose(aux[0].asnumpy(), expect_aux)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_value_and_grad_with_weights_has_aux_pynative():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad with different weights and has_aux in pynative mode.
    Expectation: No exception.
    """

    class ParamNetMultipleOutputs(nn.Cell):
        def __init__(self):
            super(ParamNetMultipleOutputs, self).__init__()
            self.w1 = Parameter(Tensor([2., 2.], mstype.float32), name="w1")
            self.w2 = Parameter(Tensor([3., 3.], mstype.float32), name="w2")

        def construct(self, x):
            res = x * self.w1 * self.w2
            return res, x, self.w1

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetMultipleOutputs()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    expect_value0 = np.array([6, 12]).astype(np.float32)
    expect_value1 = np.array([1, 2]).astype(np.float32)
    expect_value2 = np.array([2, 2]).astype(np.float32)
    value, gradient = value_and_grad(net, 0, weights, True)(x)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(value[2].asnumpy(), expect_value2)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1].asnumpy(), expect_grad_weight2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_value_and_grad_nest_with_weights_pynative():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad with different weights and has_aux as well as nested nets in pynative mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([24, 24]).astype(np.float32)
    expect_grad_weight1 = np.array([12, 24]).astype(np.float32)
    expect_grad_weight2 = np.array([8, 16]).astype(np.float32)
    expect_value0 = np.array([18, 36]).astype(np.float32)
    expect_value1 = np.array([6, 12]).astype(np.float32)
    value, gradient = value_and_grad(net, 0, weights, False)(x)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1].asnumpy(), expect_grad_weight2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_value_and_grad_nest_with_weights_has_aux_pynative():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad with different weights and has_aux as well as nested nets in pynative mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super(Net, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([18, 18]).astype(np.float32)
    expect_grad_weight1 = np.array([9, 18]).astype(np.float32)
    expect_grad_weight2 = np.array([6, 12]).astype(np.float32)
    expect_value0 = np.array([18, 36]).astype(np.float32)
    expect_value1 = np.array([6, 12]).astype(np.float32)
    value, gradient = value_and_grad(net, 0, weights, True)(x)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1].asnumpy(), expect_grad_weight2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_single_output_cell_graph_with_return_ids_pynative():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with single input and single output net and output ids in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_grad = (0, Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32)))
    real_grad = grad(net, return_ids=True)(x)
    assert np.allclose(real_grad[1].asnumpy(), expect_grad[1].asnumpy())
    assert np.allclose(real_grad[0], expect_grad[0])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_multiple_outputs_cell_graph_with_return_ids_pynative():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with single input and multiple outputs net and output ids in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputsNet()
    expect_grad = (0, Tensor(np.array([[5, 14], [29, 50]]).astype(np.float32)))
    real_grad = grad(net, return_ids=True)(x)
    assert np.allclose(real_grad[1].asnumpy(), expect_grad[1].asnumpy())
    assert np.allclose(real_grad[0], expect_grad[0])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_single_output_cell_graph_with_return_ids_pynative():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with multiple inputs and single output net and output ids in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    expect_grad1 = (1, Tensor(np.array([[0, 6], [15, -4]]).astype(np.float32)))
    expect_grad2 = (2, Tensor(np.array([[-2, 6], [-3, 8]]).astype(np.float32)))
    real_grad = grad(net, grad_position=(1, 2), return_ids=True)(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0][1].asnumpy(), expect_grad1[1].asnumpy())
    assert np.allclose(real_grad[1][1].asnumpy(), expect_grad2[1].asnumpy())
    assert np.allclose(real_grad[0][0], expect_grad1[0])
    assert np.allclose(real_grad[1][0], expect_grad2[0])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_multiple_outputs_cell_graph_with_return_ids_pynative():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with multiple inputs and multiple outputs net and output ids in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsMultipleOutputsNet()
    expect_grad1 = (
        1, Tensor(np.array([[-4, 12], [13, 0]]).astype(np.float32)))
    expect_grad2 = (2, Tensor(np.array([[-2, 12], [7, 6]]).astype(np.float32)))
    real_grad = grad(net, grad_position=(1, 2), return_ids=True)(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0][1].asnumpy(), expect_grad1[1].asnumpy())
    assert np.allclose(real_grad[1][1].asnumpy(), expect_grad2[1].asnumpy())
    assert np.allclose(real_grad[0][0], expect_grad1[0])
    assert np.allclose(real_grad[1][0], expect_grad2[0])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_iteration_function_graph_with_return_ids_pynative():
    """
    Features: Function grad_with_ids.
    Description: Test calling F.grad iterative with function and output ids in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad1 = (
        1, Tensor(np.array([[0, 12], [30, -8]]).astype(np.float32)))
    expect_grad2 = (
        2, Tensor(np.array([[-4, 12], [-6, 16]]).astype(np.float32)))
    real_grad = grad(grad(iteration_grad_function),
                     grad_position=(1, 2), return_ids=True)(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0][1].asnumpy(), expect_grad1[1].asnumpy())
    assert np.allclose(real_grad[1][1].asnumpy(), expect_grad2[1].asnumpy())
    assert np.allclose(real_grad[0][0], expect_grad1[0])
    assert np.allclose(real_grad[1][0], expect_grad2[0])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_with_weights_with_return_ids_pynative():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with different weights and output ids in pynative mode.
    Expectation: No exception.
    """

    class ParamNetRI(nn.Cell):
        def __init__(self):
            super(ParamNetRI, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            res = x * self.w * self.z
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetRI()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = (0, np.array([7, 7]).astype(np.float32))
    expect_grad_weight1 = np.array([4, 7]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    res = grad(net, 0, weights, return_ids=True)(x)
    assert np.allclose(res[0][1].asnumpy(), expect_grad_input[1])
    assert np.allclose(res[1][0][1].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[1][1][1].asnumpy(), expect_grad_weight2)
    assert np.allclose(res[0][0], expect_grad_input[0])
    assert res[1][0][0] == weights[0].name
    assert res[1][1][0] == weights[1].name


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_jit_function_grad_with_weights_return_ids():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with different weights and output ids in pynative mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super(ParamMultipleInputNet, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    net = ParamMultipleInputNet()
    weights = net.trainable_params()

    @jit
    def user_fn(x, y):
        res = grad(net, 0, weights, return_ids=True)(x, y)
        return res

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    res = user_fn(x, y)
    expect_grad_input = (0, np.array([7, 7]).astype(np.float32))
    expect_grad_weight1 = np.array([4, 7]).astype(np.float32)
    assert np.allclose(res[0][1].asnumpy(), expect_grad_input[1])
    assert np.allclose(res[1][0][1].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[0][0], expect_grad_input[0])
    assert res[1][0][0] == weights[0].name


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_construct_grad_with_weights_with_return_ids():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with different weights and output ids in pynative mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super(ParamMultipleInputNet, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, 0, self.weights, return_ids=True)(x, y)
            return res, self.weights

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet()
    grad_net = GradNet(inner_net)
    res, weights = grad_net(x, y)
    expect_grad_input = (0, np.array([7, 7]).astype(np.float32))
    expect_grad_weight1 = np.array([4, 7]).astype(np.float32)
    assert np.allclose(res[0][1].asnumpy(), expect_grad_input[1])
    assert np.allclose(res[1][0][1].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[0][0], expect_grad_input[0])
    assert res[1][0][0] == weights[0].name


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_grad_by_position_pynative():
    """
    Features: Function get_grad.
    Description: Test get_grad with position id and output gradient in pynative mode.
    Expectation: No exception.
    """

    class ParamNetRI(nn.Cell):
        def __init__(self):
            super(ParamNetRI, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            res = x * self.w * self.z
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetRI()
    weights = net.trainable_params()
    expect_grad_input = np.array([7, 7]).astype(np.float32)
    res = grad(net, 0, weights, return_ids=True)(x)
    grad_out = get_grad(res, 0)
    assert np.allclose(grad_out.asnumpy(), expect_grad_input)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_grad_with_parameter_pynative():
    """
    Features: Function get_grad.
    Description: Test get_grad with parameter and output gradient in pynative mode.
    Expectation: No exception.
    """

    class ParamNetRI(nn.Cell):
        def __init__(self):
            super(ParamNetRI, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            res = x * self.w * self.z
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetRI()
    weights = net.trainable_params()
    expect_grad_weight1 = np.array([4, 7]).astype(np.float32)
    res = grad(net, 0, weights, return_ids=True)(x)
    grad_out = get_grad(res, net.w)
    assert np.allclose(grad_out.asnumpy(), expect_grad_weight1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_grad_not_found_pynative():
    """
    Features: Function get_grad.
    Description: Test get_grad with invalid id and raise error in pynative mode.
    Expectation: No exception.
    """

    class ParamNetRI(nn.Cell):
        def __init__(self):
            super(ParamNetRI, self).__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            res = x * self.w * self.z
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetRI()
    weights = net.trainable_params()
    res = grad(net, 0, weights, return_ids=True)(x)
    with pytest.raises(RuntimeError):
        get_grad(res, 1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_outer_list_weight():
    """
    Features: Function get_grad.
    Description: Test get_grad with one weight and no position in pynative mode.
    Expectation: No exception.
    """
    class InnerNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter([1, 2], name='w')
            self.b = Parameter([1, 2], name='b')

        def construct(self, x, y):
            out = self.w * x + self.b + y
            return out

    class GradNet(nn.Cell):
        def __init__(self, net, pos, param, get):
            super().__init__()
            self.net = net
            self.pos = pos
            self.param = param
            self.get = get

        def construct(self, x, y):
            grad_net = grad(self.net, self.pos, self.param, return_ids=True)
            out_grad = grad_net(x, y)
            out = get_grad(out_grad, self.net.w)
            return out

    net = InnerNet()
    grad_net = GradNet(net, (0, 1), (net.w, net.b), (0, net.w))
    x = Tensor([2, 2], mstype.float32)
    y = Tensor([1, 2], mstype.float32)
    out = grad_net(x, y)
    expect_value = Tensor([2, 2], mstype.int64)
    assert np.allclose(out[0].asnumpy(), expect_value.asnumpy())
