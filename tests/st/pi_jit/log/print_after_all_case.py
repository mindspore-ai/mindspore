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
"""Helper script for pijit print_after_all log tests (migrated)."""

import numpy as np

from mindspore import jit, ops
from mindspore.common import Tensor
from mindspore.graph._utils.pijit_context import PIJitCaptureContext
from mindspore.nn import Cell

from tests.st.pi_jit.share.grad import GradOfFirstInput
from tests.st.pi_jit.share.utils import match_array


class _GraphSplitBase(Cell):
    def __init__(self):
        super().__init__()
        self.threshold = 2

    @staticmethod
    def _add_numpy_tensor(x):
        return Tensor(x.asnumpy() * 2)

    def _body(self, x):
        out = x
        for i in range(4):
            if i < self.threshold:
                out = out + self._add_numpy_tensor(x)
            else:
                out = out - x
        return out

    def construct(self, x):
        return self._body(x)


class GraphSplitPrintAfterAllNet(_GraphSplitBase):
    @jit(capture_mode="bytecode")
    @PIJitCaptureContext(jit_config={"print_after_all": True})
    def construct(self, x):
        return self._body(x)


def test_print_after_all_graph_split_for():
    """
    Feature: print_after_all logging for graph break in loop.
    Description: Enable print_after_all and execute the graph split helper to trigger traceback bytecode dump.
    Expectation: Log output contains traceback bytecode dump information.
    Migrated from: test_pijit_print_after_all.py::test_pijit_print_after_all_graph_split_for
    """
    x_np = np.ones((2, 3), np.float32)
    input_tensor = Tensor(x_np)
    grad_tensor = Tensor(np.random.randn(*x_np.shape).astype(np.float32))

    pynative_net = _GraphSplitBase()
    pynative_net.set_grad()
    pynative_output = pynative_net(input_tensor)

    jit_net = GraphSplitPrintAfterAllNet()
    jit_net.set_grad()
    jit_output = jit_net(input_tensor)

    match_array(pynative_output, jit_output)

    pynative_grad_net = GradOfFirstInput(pynative_net, sens_param=True)
    pynative_grad_net.set_train()
    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()

    pynative_grad = pynative_grad_net(input_tensor, grad_tensor)
    jit_grad = jit_grad_net(input_tensor, grad_tensor)
    match_array(pynative_grad, jit_grad, error=5)


class _TryFinallyBase(Cell):
    def __init__(self):
        super().__init__()
        self.reduce = ops.ReduceMean(keep_dims=False)

    def _body(self, x):
        try:
            out = self.reduce(x, 0)
        except ValueError:
            x = x.expand_dims(3)
            out = self.reduce(x, 3)
        else:
            out = out + out
        finally:
            out = out * out
            _ = out.asnumpy()
        return out

    def construct(self, x):
        return self._body(x)


class TryFinallyPrintAfterAllNet(_TryFinallyBase):
    @jit(capture_mode="bytecode")
    @PIJitCaptureContext(jit_config={"print_after_all": True})
    def construct(self, x):
        return self._body(x)


def test_print_after_all_try_finally_break():
    """
    Feature: print_after_all logging for try/finally graph break.
    Description: Enable print_after_all and execute the try/finally helper to dump codegen bytecode.
    Expectation: Log output contains one-stage bytecode collection and final bytecode dump.
    Migrated from: test_pijit_print_after_all.py::test_pijit_print_after_all_try_finally_break
    """
    x_np = np.ones((2, 3, 4), np.float32)
    input_tensor = Tensor(x_np)
    out_shape = (3, 4)
    grad_tensor = Tensor(np.random.randn(*out_shape).astype(np.float32))

    pynative_net = _TryFinallyBase()
    pynative_net.set_grad()
    pynative_output = pynative_net(input_tensor)

    jit_net = TryFinallyPrintAfterAllNet()
    jit_net.set_grad()
    jit_output = jit_net(input_tensor)

    match_array(pynative_output, jit_output)

    pynative_grad_net = GradOfFirstInput(pynative_net, sens_param=True)
    pynative_grad_net.set_train()
    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()

    pynative_grad = pynative_grad_net(input_tensor, grad_tensor)
    jit_grad = jit_grad_net(input_tensor, grad_tensor)
    match_array(pynative_grad, jit_grad, error=5)


class _BeforePsJitBase(Cell):
    def __init__(self):
        super().__init__()
        self.scale = 1

    def func1(self, x):
        return ops.square(x) * self.scale

    def func2(self, x):
        return ops.sin(x) * self.scale

    def construct(self, x):
        y = self.func1(x)
        z = self.func2(y)
        return ops.cos(z)


class BeforePsJitPrintAfterAllNet(_BeforePsJitBase):
    @jit(capture_mode="bytecode")
    @PIJitCaptureContext(jit_config={"print_after_all": True})
    def func1(self, x):
        return ops.square(x) * self.scale

    @jit(capture_mode="ast")
    def func2(self, x):
        return ops.sin(x) * self.scale


def test_print_after_all_before_psjit():
    """
    Feature: print_after_all logging before psjit execution.
    Description: Enable print_after_all with a mixed bytecode/ast pipeline to verify codegen bytecode dumps.
    Expectation: Log output contains one-stage bytecode collection and final bytecode dump.
    Migrated from: test_pijit_print_after_all.py::test_pijit_print_after_all_before_psjit
    """
    x_np = np.ones((2, 3), np.float32)
    input_tensor = Tensor(x_np)

    pynative_net = _BeforePsJitBase()
    pynative_output = pynative_net(input_tensor)

    jit_net = BeforePsJitPrintAfterAllNet()
    jit_output = jit_net(input_tensor)

    match_array(pynative_output, jit_output)
