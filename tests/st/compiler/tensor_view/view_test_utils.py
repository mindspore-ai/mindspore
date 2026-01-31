# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test tensor view utils."""
import numpy as np
import torch
import os
import shutil
from mindspore import Tensor
from mindspore.common import dtype as mstype
from tests.st.pynative.utils import allclose_nparray

def check_ir_view_op(dir_path, target_op_name):
    """Check ir if exist view op"""
    found_target = False
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"'{dir_path}' is not a valid directory.")

    for filename in os.listdir(dir_path):
        if 'validate' in filename:
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if target_op_name in content:
                    found_target = True

    assert found_target, "View op not found in IR!"
    shutil.rmtree(dir_path)

class TensorViewTestModule:
    def __init__(self, op_name, input_args, op_kwargs=None):
        """
        op_name: str, name of the operator (e.g. 'chunk', 'split')
        input_args: list, arguments for the operator. Assumes input_args[0] is the input tensor.
        op_kwargs: dict, keyword arguments.
        """
        self.ms_dtype = input_args[0].dtype
        self.op_name = op_name
        self.input_args = input_args
        self.op_kwargs = op_kwargs if op_kwargs else {}
        self.inputs_with_grad_torch = []
        self.input_x_np = input_args[0].asnumpy()
        self.output_grad_np = None # Cache random gradient

        if self.ms_dtype == mstype.float16:
            self.loss = 1e-3
        elif self.ms_dtype == mstype.float32:
            self.loss = 1e-4
        elif self.ms_dtype == mstype.bfloat16:
            self.loss = 4e-3
        elif self.ms_dtype == mstype.float64:
            self.loss = 1e-5
        else:
            self.loss = 0
        self.supports_grad = self.ms_dtype in (mstype.float16, mstype.float32, mstype.float64, mstype.bfloat16)

    def forward_pytorch_impl(self):
        if self.input_x_np.dtype == np.uint16:
            x_np = self.input_x_np.astype(np.int16)
        elif self.input_x_np.dtype == np.uint32:
            x_np = self.input_x_np.astype(np.int32)
        elif self.input_x_np.dtype == np.uint64:
            x_np = self.input_x_np.astype(np.int64)
        elif self.input_x_np.dtype in (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                       np.uint32, np.uint64, np.float16, np.float32, np.float64, np.complex64,
                                       np.complex128):
            x_np = self.input_x_np
        else:
            x_np = self.input_x_np.astype(np.float32)

        x = torch.from_numpy(x_np)

        # BFloat16 handling
        if self.ms_dtype == mstype.bfloat16:
            x = x.type(torch.bfloat16)

        rest_args = self.input_args[1:]
        op = getattr(x, self.op_name)
        out = op(*rest_args, **self.op_kwargs)

        # Output conversion aligned with reference
        if self.ms_dtype == mstype.bfloat16:
            if isinstance(out, (tuple, list)):
                return [t.detach().float().numpy() for t in out]
            return out.detach().float().numpy()

        if isinstance(out, (tuple, list)):
            return [t.detach().numpy() for t in out]
        return out.detach().numpy()

    def backward_pytorch_impl(self):
        # 1. Re-prepare Input
        if self.input_x_np.dtype in (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                     np.uint32, np.uint64, np.float16, np.float32, np.float64, np.complex64,
                                     np.complex128):
            x_np = self.input_x_np
        else:
            x_np = self.input_x_np.astype(np.float32)

        x = torch.from_numpy(x_np)
        if self.ms_dtype == mstype.bfloat16:
            x = x.type(torch.bfloat16)
        x.requires_grad = True
        self.inputs_with_grad_torch = [x]

        # 2. Re-Run Forward
        rest_args = self.input_args[1:]
        op = getattr(x, self.op_name)
        out = op(*rest_args, **self.op_kwargs)

        # 3. Generate or Use Cached Random Sens
        if self.output_grad_np is None:
            if isinstance(out, (tuple, list)):
                if self.ms_dtype == mstype.bfloat16:
                    self.output_grad_np = [np.random.randn(*t.shape).astype(np.float32) for t in out]
                else:
                    # Handle 0-dim tensor output from unbind which causes randn to return float instead of ndarray
                    self.output_grad_np = []
                    for t in out:
                        grad = np.random.randn(*t.shape)
                        # If grad is a scalar float (from 0-d shape), convert to array
                        if isinstance(grad, (float, int)):
                            grad = np.array(grad)
                        self.output_grad_np.append(grad.astype(t.detach().numpy().dtype))
            else:
                t = out.detach().numpy()
                if self.ms_dtype == mstype.bfloat16:
                    # For bfloat16, use float32 in numpy
                    self.output_grad_np = np.random.randn(*t.shape).astype(np.float32)
                else:
                    self.output_grad_np = np.random.randn(*t.shape).astype(t.dtype)

        # 4. Backward with Sens
        if isinstance(out, (tuple, list)):
            output_grads = [torch.tensor(g, dtype=t.dtype) for g, t in zip(self.output_grad_np, out)]
            torch.autograd.backward(out, output_grads)
        else:
            output_grad = torch.tensor(self.output_grad_np, dtype=out.dtype)
            out.backward(output_grad)

        # 5. Get Gradients
        grads = []
        for t in self.inputs_with_grad_torch:
            if self.ms_dtype == mstype.bfloat16:
                grads.append(t.grad.detach().float().numpy())
            else:
                grads.append(t.grad.detach().numpy())
        return grads

    def forward_cmp(self, ms_forward_func):
        out_torch_np = self.forward_pytorch_impl()
        out_ms = ms_forward_func(*self.input_args, **self.op_kwargs)

        if self.ms_dtype == mstype.bfloat16:
            if isinstance(out_ms, (tuple, list)):
                out_ms_np = [t.float().asnumpy() for t in out_ms]
            else:
                out_ms_np = out_ms.float().asnumpy()
        else:
            if isinstance(out_ms, (tuple, list)):
                out_ms_np = [t.asnumpy() for t in out_ms]
            else:
                out_ms_np = out_ms.asnumpy()

        if isinstance(out_torch_np, (tuple, list)):
            assert isinstance(out_ms_np, (tuple, list)), "Output type mismatch"
            assert len(out_torch_np) == len(out_ms_np), "Output length mismatch"
            for t, m in zip(out_torch_np, out_ms_np):
                allclose_nparray(t, m, self.loss, self.loss)
        else:
            allclose_nparray(out_torch_np, out_ms_np, self.loss, self.loss)

    def backward_cmp(self, ms_backward_func):
        if not self.supports_grad:
            return
        # 4. PyTorch Backward
        # Note: backward_pytorch_impl now self-contained re-runs forward to build graph
        grads_torch = self.backward_pytorch_impl()

        # Prepare Sens Tensor
        if isinstance(self.output_grad_np, (list, tuple)):
            if self.ms_dtype == mstype.bfloat16:
                sens = tuple(Tensor(g, mstype.bfloat16) for g in self.output_grad_np)
            else:
                sens = tuple(Tensor(g) for g in self.output_grad_np)
        else:
            if self.ms_dtype == mstype.bfloat16:
                sens = Tensor(self.output_grad_np, mstype.bfloat16)
            else:
                sens = Tensor(self.output_grad_np)

        grads_ms = ms_backward_func(*self.input_args, sens, **self.op_kwargs)

        if not isinstance(grads_ms, (tuple, list)):
            grads_ms_np = [grads_ms.asnumpy()]
            if self.ms_dtype == mstype.bfloat16:
                grads_ms_np = [grads_ms.float().asnumpy()]
        else:
            if self.ms_dtype == mstype.bfloat16:
                grads_ms_np = [g.float().asnumpy() for g in grads_ms]
            else:
                grads_ms_np = [g.asnumpy() for g in grads_ms]

        if len(grads_torch) == 0:
            return

        if len(grads_ms_np) >= 1 and len(grads_torch) >= 1:
            allclose_nparray(grads_torch[0], grads_ms_np[0], self.loss, self.loss)
