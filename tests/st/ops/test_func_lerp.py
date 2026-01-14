# Copyright 2024 Huawei Technocasties Co., Ltd
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

import mindspore as ms
from mindspore import mint
import numpy as np
import pytest
import torch

from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.pynative.utils import GradOfAllInputs, allclose_nparray
from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def lerp_forward_func(x, y, z):
    return mint.lerp(x, y, z)


class LerpNet(ms.nn.Cell):
    def construct(self, x, y, z):
        return mint.lerp(x, y, z)


class TorchLerpNet(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.lerp(x, y, z)


class TestLerpModule():
    def __init__(self, inputs=None):
        self.ms_dtype = inputs[0].dtype
        self.input_x = inputs[0]
        self.input_y = inputs[1]
        self.input_z = inputs[2]
        if isinstance(inputs[0], ms.Tensor):
            self.input_x_np = inputs[0].asnumpy()
        else:
            self.input_x_np = np.array(inputs[0], dtype=np.float32)
        if isinstance(inputs[1], ms.Tensor):
            self.input_y_np = inputs[1].asnumpy()
        else:
            self.input_y_np = np.array(inputs[1], dtype=np.float32)
        if isinstance(inputs[2], ms.Tensor):
            self.input_z_np = inputs[2].asnumpy()
        else:
            self.input_z_np = np.array(inputs[2], dtype=np.float32)

        if self.ms_dtype == ms.float16:
            self.loss = 1e-3
        elif self.ms_dtype in (ms.float32, ms.complex64):
            self.loss = 1e-4
        elif self.ms_dtype in (ms.float64, ms.complex128):
            self.loss = 1e-5
        elif self.ms_dtype == ms.bfloat16:
            self.loss = 4e-3
        else:
            self.loss = 0
        self.out_grad_np = None

    def forward_mindspore_impl(self):
        net = LerpNet()
        out = net(self.input_x, self.input_y, self.input_z)
        if out.dtype == ms.bfloat16:
            return out.float().asnumpy()
        return out.asnumpy()

    def grad_mindspore_impl(self):
        if self.out_grad_np is None:
            out = self.forward_mindspore_impl()
            sens = np.random.randn(*list(out.shape))
            if isinstance(sens, float):
                self.out_grad_np = sens
            else:
                self.out_grad_np = sens.astype(dtype=out.dtype)
        if self.ms_dtype == ms.bfloat16:
            ms_output_grad = ms.Tensor(self.out_grad_np, ms.bfloat16)
        else:
            ms_output_grad = ms.Tensor(self.out_grad_np)
        net = LerpNet()
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        grads = grad_net(self.input_x, self.input_y, self.input_z, ms_output_grad)
        grads_np = []
        for g in grads:
            if g is None:
                grads_np.append(None)
            elif g.dtype == ms.bfloat16:
                grads_np.append(g.float().asnumpy())
            else:
                grads_np.append(g.asnumpy())
        return grads_np

    def forward_torch_impl(self):
        if self.ms_dtype == ms.bfloat16:
            x = torch.from_numpy(self.input_x_np.astype(np.float32)).bfloat16()
            y = torch.from_numpy(self.input_y_np.astype(np.float32)).bfloat16()
            z = torch.from_numpy(self.input_z_np.astype(np.float32)).bfloat16()
        else:
            x = torch.from_numpy(self.input_x_np)
            y = torch.from_numpy(self.input_y_np)
            z = torch.from_numpy(self.input_z_np)
        net = TorchLerpNet()
        out = net(x, y, z)
        if self.ms_dtype == ms.bfloat16:
            return out.detach().float().numpy()
        return out.detach().numpy()

    def grad_torch_impl(self):
        if self.ms_dtype == ms.bfloat16:
            x = torch.from_numpy(self.input_x_np.astype(np.float32)).bfloat16()
            y = torch.from_numpy(self.input_y_np.astype(np.float32)).bfloat16()
            z = torch.from_numpy(self.input_z_np.astype(np.float32)).bfloat16()
        else:
            x = torch.from_numpy(self.input_x_np)
            y = torch.from_numpy(self.input_y_np)
            z = torch.from_numpy(self.input_z_np)
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True
        net = TorchLerpNet()
        out = net(x, y, z)
        output_grad = torch.tensor(self.out_grad_np, dtype=out.dtype)
        out.backward(output_grad)
        if self.ms_dtype == ms.bfloat16:
            return [x.grad.detach().float().numpy(), y.grad.detach().float().numpy(), z.grad.detach().float().numpy()]
        return [x.grad.detach().numpy(), y.grad.detach().numpy(), z.grad.detach().numpy()]

    def forward_cmp(self):
        out_me = self.forward_mindspore_impl()
        out_torch = self.forward_torch_impl()
        allclose_nparray(out_torch, out_me, self.loss, self.loss)

    def grad_cmp(self):
        out_me = self.grad_mindspore_impl()
        out_torch = self.grad_torch_impl()
        for me, th in zip(out_me, out_torch):
            allclose_nparray(th, me, self.loss, self.loss)


def set_mode(mode):
    if mode == "GRAPH_MODE":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard",
          essential_mark="essential", mem_peak=2.00)
@pytest.mark.parametrize("context_mode", ["PYNATIVE_MODE", "GRAPH_MODE"])
def test_mint_lerp_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function lerp forward.
    Expectation: expect correct result.
    """
    set_mode(context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    z = generate_random_input((2, 3, 4), np.float32)
    module = TestLerpModule(inputs=[ms.Tensor(x), ms.Tensor(y), ms.Tensor(z)])
    module.forward_cmp()
    module.grad_cmp()

    # test scalar input
    module = TestLerpModule(inputs=[ms.Tensor(x), ms.Tensor(y), 0.5])
    module.forward_cmp()
    module.grad_cmp()

    # test broadcast
    x = generate_random_input((1, 3, 4), np.float32)
    y = generate_random_input((1, 3, 4), np.float32)
    z = generate_random_input((2, 2, 3, 4), np.float32)
    module = TestLerpModule(inputs=[ms.Tensor(x), ms.Tensor(y), ms.Tensor(z)])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
def test_mint_lerp_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function lerp forward with dynamic shape.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    input2 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    input3 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))

    input4 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    input5 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    input6 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    TEST_OP(lerp_forward_func,
            [[input1, input2, input3], [input4, input5, input6]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'all_dim_zero': True})

    TEST_OP(lerp_forward_func, [[input1, input2, 0.5], [input4, input5, 0.7]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'all_dim_zero': True})
