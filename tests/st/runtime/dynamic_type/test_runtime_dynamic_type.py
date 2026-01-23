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
import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter, jit
from mindspore.common import ParameterTuple, mutable
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
import mindspore.context as context
import mindspore.numpy as mnp
import mindspore.ops.functional as F
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_value_depend_infer():
    """
    Feature: Dynamic type.
    Description: Value depend in any type.
    Expectation: Not throw exception.
    """
    class ValueDependInferNet(nn.Cell):
        def __init__(self, tau=1, hard=False):
            super().__init__()
            self.tau = tau
            self.hard = hard
            self.zero = Tensor(0, ms.float32)
            self.one = Tensor(1, ms.float32)

        def construct(self, logits: Tensor):
            eps = 1e-10
            dim = -1
            U = ops.uniform(logits.shape, self.zero, self.one)
            noise = -ops.log(-ops.log(U + eps) + eps)
            y = logits + noise
            y_soft = ops.Softmax()(y / self.tau)
            if self.hard:
                index = y_soft.argmax(dim)
                y_hard = ops.OneHot()(index, 2, self.one, self.zero)
                ret = y_hard
            else:
                ret = y_soft
            return ret

    class EyeNet(nn.Cell):
        def __init__(self, dtype=ms.float32):
            super(EyeNet, self).__init__()
            self.dtype = dtype
            self.insert_graph = ValueDependInferNet()

        def construct(self, x):
            def update_shape(baize_out, ex_shape):
                if isinstance(baize_out, (list, tuple)):
                    baize_out = baize_out[0]
                if baize_out.shape == ex_shape:
                    return baize_out
                out_size = baize_out.numel()
                ex_size = 1
                for dim in ex_shape:
                    ex_size *= dim
                if out_size > ex_size:
                    return ops.flatten(baize_out, start_dim=0)[:ex_size].reshape(ex_shape)
                if out_size < ex_size:
                    return ops.pad(ops.flatten(baize_out, start_dim=0),
                                   Tensor([0, ex_size - out_size]), mode='constant',
                                   value=0.1).reshape(ex_shape)
                return baize_out.reshape(ex_shape)

            baize_out = ops.eye(x.shape[1], 3, dtype=self.dtype)
            logits = update_shape(baize_out, (64, 64))
            baize_out = self.insert_graph(logits)
            return baize_out

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    net = EyeNet()
    input_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_dyn)
    x = Tensor(mnp.ones([3, 3]), dtype=ms.float32)
    out = net(x).asnumpy()
    assert out.shape == (64, 64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_arg_to_dynamic_tuple_and_partial_para():
    """
    Feature: Dynamic type.
    Description: Value depend in any type.
    Expectation: Not throw exception.
    """
    import numpy as np
    class Net(nn.Cell):
        def construct(self, list_in):
            length = len(list_in)
            if length >= 2:
                ele1 = list_in[0]
                ele2 = list_in[length - 1]
                tmp = ops.add(ele1, ele2)
                return (ele1, ele2, tmp)
            add = ops.add(list_in[0], 1)
            return (list_in[0], add)

    input1 = np.random.rand(2, 2).astype(np.float32)
    input2 = np.random.rand(2, 2).astype(np.float32)
    inputx = ms.mutable((Tensor(input1), Tensor(input2)), dynamic_len=True)
    gradnet = ms.ops.GradOperation(get_all=True)(Net())
    _ = gradnet(inputx)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_twice_execute_for_dynamic_type_graph():
    """
    Feature: Dynamic type.
    Description: Value depend in any type.
    Expectation: Not throw exception.
    """
    import numpy as np
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.l = (1, 2, 3)

        def construct(self, x, a, b):
            out = x
            while a < b:
                if 2 * a >= b:
                    for _ in range(3):
                        out = self.relu(out)
                else:
                    for _ in range(1):
                        out = divmod(out.asnumpy(), 2)
                    out = Tensor(out[1])
                a += 1
            return out

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(x)
    a = Tensor(2, ms.float32)
    b = Tensor(6, ms.float32)
    net = Net()
    net(input_x, a, b)



@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cpu_optimize_fp16():
    """
    Feature: Dynamic type.
    Description: Value depend in any type.
    Expectation: Not throw exception.
    """
    import numpy as np
    class ApplyRMSNet(nn.Cell):
        def __init__(self):
            super(ApplyRMSNet, self).__init__()
            self.apply_rms = P.ApplyRMSProp()
            self.lr = 0.001
            self.rho = 0.0
            self.momentum = 0.0
            self.epsilon = 1e-10
            self.ms = Parameter(Tensor(np.random.rand(3, 3).astype(np.float16)), name="ms")
            self.moment = Parameter(Tensor(np.random.rand(3, 3).astype(np.float16)), name="moment")

        def construct(self, var, grad):
            out = self.apply_rms(var, self.ms, self.moment, self.lr, grad, self.rho, self.momentum, self.epsilon)
            return out

    x = Tensor(np.random.rand(3, 3).astype(np.float16))
    var_value1 = Tensor(np.random.rand(3, 3).astype(np.float16))
    var_value2 = var_value1.copy()
    print(var_value1)
    print(var_value2)
    context.set_context(device_target="CPU")
    context.set_context(mode=context.GRAPH_MODE)
    var1 = Parameter(var_value1, name="var")
    net1 = ApplyRMSNet()
    net1(var1, x)
    context.set_context(mode=context.PYNATIVE_MODE)
    var2 = Parameter(var_value2, name="var")
    net2 = ApplyRMSNet()
    net2(var2, x)
    print(var1.value())
    print(var2.value())
    assert np.allclose(var1.value().asnumpy(), var2.value().asnumpy(), 1.0e-4, 1.0e-4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cpu_empty_tuple():
    """
    Feature: tensor data.
    Description: empty value sequence.
    Expectation: Not throw exception.
    """
    class _Grad(nn.Cell):
        def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
            super().__init__()
            self.network = network
            self.grad = grad
            self.sens_param = self.grad.sens_param
            self.wrt_params = wrt_params
            self.real_inputs_count = real_inputs_count

        def construct(self, *inputs):
            if self.real_inputs_count is None or self.sens_param is False:
                if self.wrt_params:
                    return self.grad(self.network, self.params)(*inputs)
                return self.grad(self.network)(*inputs)

            real_inputs = inputs[:self.real_inputs_count]
            sense_param_inputs = inputs[self.real_inputs_count:]
            if self.wrt_params:
                return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
            return self.grad(self.network)(*real_inputs, sense_param_inputs)

    class GradOfAllInputs(_Grad):
        def __init__(self, network, sens_param=True, real_inputs_count=None):
            super().__init__(grad=ops.GradOperation(get_all=True, sens_param=sens_param),
                             network=network, real_inputs_count=real_inputs_count)

    def half_fn_1(cell, inputs, outputs):
        print("inputs[0]", inputs[0])
        print("inputs[1]", inputs[1])
        return inputs[0], inputs[1]

    class EyeLayer(nn.Cell):
        def construct(self, *args, **kwargs):
            return ops.eye(*args, **kwargs)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = nn.ReLU()
            self.eye = EyeLayer()
            self.handle = self.eye.register_forward_hook(half_fn_1)

        def construct(self, x, y):
            x = self.relu(x)
            x = self.eye(x.shape[0], y)
            return x, y

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    import numpy as np
    ms_net = Net()
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2 = None
    input1_ms = Tensor(input1_np)
    ms_net.set_grad()
    out_ms = ms_net(input1_ms, input2)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    grad_net(input1_ms, input2, out_ms)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cpu_any_type_empty_tuple():
    """
    Feature: tensor data.
    Description: empty value sequence.
    Expectation: Not throw exception.
    """
    class Simplenet(nn.Cell):
        def __init__(self, w, b):
            super().__init__()
            self.ref_x1 = Parameter(Tensor(w), name='x1')
            self.ref_x2 = Parameter(Tensor(b), name='x2')

        def construct(self, x):
            return 2 * x + 3 * self.ref_x1 + self.ref_x2

    class LossFn(nn.Cell):
        def __init__(self, fn):
            super().__init__()
            self.model = fn

        def construct(self, sample, target):
            preduction = self.model(sample)
            loss = mnp.sum(preduction) - target
            return loss

    class TrainStepNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.loss_fn = LossFn(net)
            self.weight = ParameterTuple(net.trainable_params())
            self.optim = nn.Adam(self.weight, learning_rate=0.001)
            self.grad_op = ops.GradOperation(get_by_list=True, get_all=False)

        def construct(self, batch, targets):
            loss = self.loss_fn(batch, targets)
            grad_weights = self.grad_op(self.loss_fn, self.weight)(batch, targets)
            self.optim(grad_weights)
            return loss

    class VmapNet(nn.Cell):
        def __init__(self, net_list, in_axes, out_axes):
            super().__init__()
            self.net_list = net_list
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, *x):
            vmapnet = F.vmap(self.net_list, self.in_axes, self.out_axes)
            out = vmapnet(*x)
            return out

    import numpy as np
    class VmapFactory():
        def __init__(self, net_num, axes):
            self.weight = []
            self.bias = []
            self.in_axes = axes[0]
            self.out_axes = axes[1]
            for _ in range(net_num):
                w = np.random.rand(2, 3).astype(np.float32)
                self.weight.append(w)
                b = np.random.rand(2, 3).astype(np.float32)
                self.bias.append(b)

        def single_net_train(self, net_class, x, y, train):
            infer_list = []
            net_list = []
            for w, b in zip(self.weight, self.bias):
                net = net_class(w, b)
                infer_list.append(net)
                train_net = TrainStepNet(net)
                net_list.append(train_net)
            loss_list = []
            if train:
                for i in range(2):
                    if self.in_axes[1] == 0:
                        label = y[i]
                    else:
                        label = y
                    loss = [train_net(x, label).asnumpy() for train_net in net_list]
                    loss_list.append(loss)
                out_list = [net(x).asnumpy() for net in infer_list]
                output = (np.array(out_list), np.array(loss_list))
            return output

        def vmap_net_train(self, net_class, x, y, train):
            infer_list = []
            net_list = []
            for w, b in zip(self.weight, self.bias):
                net = net_class(w, b)
                infer_list.append(net)
                train_net = TrainStepNet(net)
                net_list.append(train_net)
            net_list = nn.CellList(net_list)
            loss_list = []
            if train:
                vmap_net = VmapNet(net_list, in_axes=self.in_axes, out_axes=self.out_axes)
                for _ in range(2):
                    loss = vmap_net(x, y)
                    loss_list.append(loss.asnumpy())
                out_list = []
                for net in infer_list:
                    out = net(x)
                    out_list.append(out.asnumpy())
                output = (np.array(out_list), np.array(loss_list))
            return output

    fact = VmapFactory(net_num=3, axes=((None, None), 0))
    x = Tensor([[1, 2, 3], [2, 3, 4]], mstype.float32)
    y = 24
    a, b = fact.single_net_train(Simplenet, x, y, True)
    c, d = fact.vmap_net_train(Simplenet, x, y, True)
    print(a, b, c, d)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=4.0)
def test_pyobj_to_tensor():
    """
    Feature: tensor data.
    Description: empty value sequence.
    Expectation: Not throw exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.idx = mutable([2, 1, 0])

        def construct(self, x):
            out = x[self.idx]
            return out

    class _Grad(nn.Cell):
        def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
            super().__init__()
            self.network = network
            self.grad = grad
            self.sens_param = self.grad.sens_param
            self.wrt_params = wrt_params
            self.real_inputs_count = real_inputs_count

        def construct(self, *inputs):
            if self.real_inputs_count is None or self.sens_param is False:
                if self.wrt_params:
                    return self.grad(self.network, self.params)(*inputs)
                return self.grad(self.network)(*inputs)

            real_inputs = inputs[:self.real_inputs_count]
            sense_param_inputs = inputs[self.real_inputs_count:]
            if self.wrt_params:
                return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
            return self.grad(self.network)(*real_inputs, sense_param_inputs)

    class GradOfAllInputs(_Grad):
        def __init__(self, network, sens_param=True, real_inputs_count=None):
            super().__init__(grad=ops.GradOperation(get_all=True, sens_param=sens_param),
                             network=network, real_inputs_count=real_inputs_count)

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    import numpy as np
    x = Tensor(np.random.rand(3, 3, 2), dtype=mstype.float32)
    d = Tensor(None, dtype=mstype.float32)
    net = Net()
    net.set_inputs(d)
    out = net(x)
    grad_net = GradOfAllInputs(net, False)
    grad_out = grad_net(x)
    print(out)
    assert grad_out[0].asnumpy()[0][0][0] == 1


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=4.0)
def test_pyexecute_launch_d2h():
    """
    Feature: tensor data.
    Description: empty value sequence.
    Expectation: Not throw exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = 1

        def construct(self, y):
            return ops.matmul(self.x, y)

    class GradNet(nn.Cell):
        def __init__(self, net, grad_position=0):
            super().__init__()
            self.grad = ops.grad
            self.grad_net = self.grad(net, grad_position=grad_position)

        def construct(self, *x):
            return self.grad_net(*x)

    @jit
    def modify_func():
        obj.x = Tensor([[-1, 0], [0, -1]], ms.float32)
        obj.x += 1
        return obj.x

    import numpy as np
    obj = Net()
    ret1 = modify_func()
    y = Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
    ms_out = obj(y)
    ms_grad = GradNet(obj)(y)
    ret2 = modify_func()

    assert np.allclose(np.array([[0, 1], [1, 0]]), ret1.asnumpy(), 0, 0)
    assert np.allclose(np.array([[4, 5, 6], [1, 2, 3]]), ms_out.asnumpy(), 0, 0)
    assert np.allclose(np.ones([2, 3]), ms_grad.asnumpy(), 0, 0)
    assert np.allclose(np.array([[0, 1], [1, 0]]), obj.x.asnumpy(), 0, 0)
    assert np.allclose(ret2.asnumpy(), ret1.asnumpy(), 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_sequence_slice():
    """
    Feature: dynamic shape and dynamic value op.
    Description: sequenceslice op.
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self, list_input, flag=1):
            super().__init__()
            self.tensor_list = list_input
            self.flag = flag

        def construct(self, x, y):
            if self.flag == 1:
                start = x + y
                tensor_list_slice = self.tensor_list[start::]
            elif self.flag == 2:
                stop = x + y
                tensor_list_slice = self.tensor_list[:stop:]
            else:
                step = x + y
                tensor_list_slice = self.tensor_list[::step]
            return tensor_list_slice

    ms.runtime.dispatch_threads_num(2)
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"}, device_target="CPU")
    import numpy as np
    tensor_list = [1, 3, 4, 15, 6, 7, 9, 8, 14]
    x = Tensor(np.array(0), ms.int32)
    y = Tensor(np.array(1), ms.int32)
    net = Net(tensor_list, flag=2)
    print(net(x, y))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_pyexecute_for_single_pipeline():
    """
    Feature: dynamic shape and dynamic value op.
    Description: pyexecute op.
    Expectation: success
    """
    import numpy as np
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(Tensor(np.full((2, 2), 2), ms.float32), name="w")
            self.m = 2

        def construct(self, x):
            self.weight = x
            self.m = 3
            return x

    ms.runtime.dispatch_threads_num(2)
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"}, device_target="CPU")
    x = Tensor(np.full((2, 2), 5).astype(np.float32))
    net = Net()
    print(net(x))
