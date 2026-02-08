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
# ============================================================================

import os

import mindspore as ms
from mindspore import nn, ops, jit, Tensor, Parameter, value_and_grad
from mindspore._c_expression import PyNativeExecutor_
from mindspore.graph._utils.pijit_context import PIJitCaptureContext
from mindspore.graph.api import _JitExecutor, _PyNativeExecutor
import numpy as np


iteration = 0
param_hook_called_count = 0
ori_mul = ops.mul
data_path = './data'


def empty(self, *args, **kwargs):
    pass


def save_npy(data_type, data):
    if iteration == 0:
        rank_id = ms.communication.get_rank() if ms.communication.GlobalComm.INITED else None
    else:
        rank_id = os.environ["RANK_ID"] if ms.communication.GlobalComm.INITED else None
    file_name = f'{data_path}/rank_{rank_id}_step_{iteration}_{data_type}.npy'
    np.save(file_name, data)


def forward_pre_hook(cell, inputs):
    assert isinstance(cell, nn.Cell)
    save_npy('forward_pre', inputs[0].numpy())


def forward_hook(cell, inputs, outputs):
    assert isinstance(cell, nn.Cell)
    save_npy('forward', outputs.numpy())


def backward_pre_hook(cell, grad_out):
    assert False


def backward_hook(cell, grad_in, grad_out):
    assert isinstance(cell, nn.Cell)
    assert grad_in == (None,)
    save_npy('backward', grad_out[0].numpy())


def tensor_hook(grad):
    save_npy('tensor_grad', grad[0].numpy())


def param_hook(grad):
    global param_hook_called_count
    param_hook_called_count += 1
    assert param_hook_called_count < 2
    save_npy('param_grad', grad.numpy())
    return grad


class JitDump(_JitExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = PyNativeExecutor_.get_instance()

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        save_npy('jit_forward', out.numpy())
        return out

    def grad(self, obj, grad, weights, grad_position, *args, **kwargs):
        output = self._executor.grad(grad, obj, weights, grad_position, False, *args, *(kwargs.values()))
        save_npy('jit_backward', args[0].numpy())
        return output


class MulCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(forward_pre_hook)
        self.register_forward_hook(forward_hook)
        self.register_backward_pre_hook(backward_pre_hook)
        self.register_backward_hook(backward_hook)

    def construct(self, *args, **kwargs):
        return ori_mul(*args, **kwargs)


def wrapped_mul(*args, **kwargs):
    return MulCell()(*args, **kwargs)


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sub = ops.Sub()
        self.relu = nn.ReLU()
        self.mul_add = MulAddNet()

    def construct(self, x, y):
        z = self.sub(x, y)
        z = self.mul_add(z, x)
        z = self.relu(z)
        z = ops.HookBackward(tensor_hook)(z)
        z = ops.mul(z, 0.5)
        z.add_(1.0)
        output = z * 4.0
        return output


class MulAddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.bias = Parameter(Tensor([-1.0], ms.float32), name="bias")

    @jit
    def construct(self, x, y):
        z = ops.matmul(x, y)
        z = z + self.bias
        return z


def run_net(dump_path):
    global iteration, data_path
    assert os.path.isdir(dump_path)
    data_path = dump_path

    ms.set_device('Ascend')
    ms.set_context(mode=ms.PYNATIVE_MODE)

    PIJitCaptureContext.__enter__ = empty
    PIJitCaptureContext.__exit__ = empty
    setattr(ms.graph.api, '_JitExecutor', JitDump)
    setattr(_PyNativeExecutor, 'grad', JitDump.grad)
    setattr(ops, 'mul', wrapped_mul)

    input_tensor0 = Tensor([[1.5, 1.5], [1.5, 1.5]])
    input_tensor1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    net = SimpleNet()
    grad_fn = value_and_grad(net, grad_position=(0, 1), weights=net.mul_add.bias)

    handle = net.mul_add.bias.register_hook(param_hook)
    grad_fn(input_tensor0, input_tensor1)
    iteration += 1
    handle.remove()
    grad_fn(input_tensor0, input_tensor1)
    iteration += 1
