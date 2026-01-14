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
# ==============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, mutable, jit, ops, mint
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_output_paramter_double():
    """
    Feature: Support tensor inplace.
    Description: Fix the input host tensor.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            z = x + y
            return z, z, x, x

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    print(net(input_x, input_y))
    print(net(input_x, input_y))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_output_cnode_double():
    """
    Feature: Support tensor inplace.
    Description: Fix the input host tensor.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            z = x + y
            return z, z

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    print(net(input_x, input_y))
    print(net(input_x, input_y))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_output_paramter_double_control_flow():
    """
    Feature: Support tensor inplace.
    Description: Fix the input host tensor.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            if x > 1:
                return x, x
            return y, y

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    print(net(input_x, input_y))
    print(net(input_x, input_y))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_no_used_input_from_copy_actor_to_super_kerenl_actor():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")
            self.depend = P.Depend()

        def construct(self, x, y):
            z1 = self.add(x, y)
            z2 = x - y
            return z1, z2

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 5
    assert out[1] == -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_no_used_input_from_super_kernel_actor_to_super_kerenl_actor():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")
            self.depend = P.Depend()

        def construct(self, x, y):
            z1 = x * y
            z2 = self.add(z1, y)
            z3 = self.add(z2, y)
            z4 = z3 - y
            return z1, z2, z3, z4

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 6
    assert out[1] == 9
    assert out[2] == 12
    assert out[3] == 9


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_no_used_input_from_parameter_store_to_super_kerenl_actor():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")

        def construct(self, x, y):
            z3 = y * y
            return x.shape, z3

    net = Net()
    x_dyn = Tensor(shape=[None], dtype=ms.int64)
    y_dyn = Tensor(shape=[None], dtype=ms.int64)
    net.set_inputs(x_dyn, y_dyn)
    input_x = ms.Tensor([2])
    input_y = ms.Tensor([3])
    out = net(input_x, input_y)
    print("out:", out)
    assert out[1] == 9


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_no_used_flag_parameter():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")

        def construct(self, x):
            return x[1] + x[1]

    net = Net()
    input_x = ms.Tensor([2])
    input_y = ms.Tensor([3])
    input_all = mutable([input_x, input_y])
    out = net(input_all)
    print("out:", out)
    assert out == 6


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_no_used_flag_parameter_2():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    @jit
    def avg_pool_forward_func(x):
        return ops.AvgPool(kernel_size=2, strides=2, pad_mode="VALID", data_format="NCHW")(x)

    @jit
    def avg_pool_backward_func(x):
        return ops.grad(avg_pool_forward_func, (0,))(x)

    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    grads = avg_pool_backward_func(x)
    print(grads)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_entrance_to_fusion_to_kernel_actor():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class NLLLossNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = mint.nn.functional.nll_loss

        def construct(self, input_x, target, weight, ignore_index, reduction):
            return self.op(input_x, target, weight, ignore_index=ignore_index, reduction=reduction)

    input_x = Tensor(None, dtype=mstype.float32)
    target = Tensor(None, dtype=mstype.int32)
    weight = Tensor(None, dtype=mstype.float32)
    ignore_index = mutable(input_data=-3, dynamic_len=False)
    reduction = 'mean'

    input1 = Tensor(np.random.randn(3, 3, 9), mstype.float32)
    target1 = Tensor(np.random.randn(3, 9), mstype.int32)
    weight1 = Tensor(np.random.randn(3,), dtype=mstype.float32)
    ignore_index1 = mutable(input_data=-3, dynamic_len=False)
    reduction1 = 'mean'

    input2 = Tensor(np.random.randn(8, 8, 7, 7, 5, 4, 5, 8), mstype.float32)
    target2 = Tensor(np.random.randn(8, 7, 7, 5, 4, 5, 8), mstype.int32)
    weight2 = Tensor(np.random.randn(8,), dtype=mstype.float32)
    ignore_index2 = mutable(input_data=0, dynamic_len=False)
    reduction2 = 'mean'

    input3 = Tensor(np.random.randn(3, 5), mstype.float32)
    target3 = Tensor(np.random.randn(3,), mstype.int32)
    weight3 = Tensor(np.random.randn(5,), dtype=mstype.float32)
    ignore_index3 = mutable(input_data=0, dynamic_len=False)
    reduction3 = 'mean'

    net = NLLLossNet()
    net.set_inputs(input_x, target, weight, ignore_index, reduction)
    out1 = net(input1, target1, weight1, ignore_index1, reduction1)
    out2 = net(input2, target2, weight2, ignore_index2, reduction2)
    out3 = net(input3, target3, weight3, ignore_index3, reduction3)
    print(out1)
    print(out2)
    print(out3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_output_ref_count_max():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class _Grad(nn.Cell):
        def __init__(self, grad, network):
            super().__init__()
            self.network = network
            self.grad = grad

        def construct(self, *inputs):
            return self.grad(self.network)(*inputs)


    class GradOfFirstInput(_Grad):
        """
        get grad of first input
        """

        def __init__(self, network, sens_param=True):
            super().__init__(grad=ops.GradOperation(sens_param=sens_param), network=network)


    class Median(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = mint.median

        def construct(self, input_x, dim, keepdim):
            return self.op(input_x, dim, keepdim)


    input_x = Tensor(shape=(None, None, None, None), dtype=mstype.float32)
    dim = mutable(input_data=-3, dynamic_len=False)
    keepdim = mutable(input_data=True, dynamic_len=False)

    input1 = Tensor(np.random.randn(5, 7, 3, 4), mstype.float32)
    dim1 = mutable(input_data=-3, dynamic_len=False)
    keepdim1 = mutable(input_data=True, dynamic_len=False)

    input2 = Tensor(np.random.randn(9, 8, 4, 3), mstype.float32)
    dim2 = mutable(input_data=-4, dynamic_len=False)
    keepdim2 = mutable(input_data=False, dynamic_len=False)

    input3 = Tensor(np.random.randn(5, 4, 8, 7), mstype.float32)
    dim3 = mutable(input_data=-3, dynamic_len=False)
    keepdim3 = mutable(input_data=False, dynamic_len=False)

    context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    net = Median()
    grad_net = GradOfFirstInput(net, sens_param=False)
    grad_net.set_inputs(input_x, dim, keepdim)
    grad_net.set_train()
    out1 = grad_net(input1, dim1, keepdim1)
    out2 = grad_net(input2, dim2, keepdim2)
    out3 = grad_net(input3, dim3, keepdim3)
    print(out1)
    print(out2)
    print(out3)

    context.set_context(mode=ms.PYNATIVE_MODE)
    pynative_net = Median()
    pynative_grad_net = GradOfFirstInput(pynative_net, sens_param=False)
    pynative_grad_net.set_inputs(input_x, dim, keepdim)
    pynative_grad_net.set_train()
    pynative_out1 = pynative_grad_net(input1, dim1, keepdim1)
    pynative_out2 = pynative_grad_net(input2, dim2, keepdim2)
    pynative_out3 = pynative_grad_net(input3, dim3, keepdim3)
    print(pynative_out1)
    print(pynative_out2)
    print(pynative_out3)

    for a, b in zip(pynative_out1, out1):
        assert np.allclose(a.asnumpy(), b.asnumpy())
    for a, b in zip(pynative_out2, out2):
        assert np.allclose(a.asnumpy(), b.asnumpy())
    for a, b in zip(pynative_out3, out3):
        assert np.allclose(a.asnumpy(), b.asnumpy())
