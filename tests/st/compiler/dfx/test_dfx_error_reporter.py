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
""" dfx test """
import numpy as np
import pytest
from mindspore.nn import Cell
from mindspore.ops import operations as op
from mindspore import Tensor, jit, context
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs
from tests.st.compiler.utils import OpsFactory


class Sort(Cell):
    def __init__(self, axis, descending):
        super().__init__()
        self.op = op.Sort(axis=axis, descending=descending)

    @jit(backend="ms_backend")
    def construct(self, x):
        return self.op(x)


class SortFactory(OpsFactory):
    def __init__(self, input_shape, axis, descending, dtype=np.float32):
        super().__init__(dtype=dtype)
        self.dtype = dtype
        self.input_x = np.random.randn(*input_shape).astype(dtype=dtype)
        self.axis = axis
        self.descending = descending
        self.output_grad_np = None
        self.output_grad_np1 = None
        self.output_grad_np2 = None

    def forward_mindspore_impl(self):
        input_me = Tensor(self.input_x)
        net = Sort(self.axis, self.descending)
        out = net(input_me)
        return out[0].asnumpy(), out[1].asnumpy()


class ReLUNetWithSortInBprop(Cell):
    def __init__(self):
        super().__init__()
        self.relu = op.ReLU()
        self.sort = op.Sort(axis=-4)

    def construct(self, x):
        return self.relu(x)

    def bprop(self, x, out, dout):
        return self.sort(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dfx_sort_exception_axis_value():
    """
    Feature: Dfx Check.
    Description: Check if the dfx meets the expected requirements.
    Expectation: Dfx meets the expected requirements.
    """
    with pytest.raises(ValueError) as error_log:
        fact = SortFactory(input_shape=(2, 3, 4), axis=-4,
                           descending=False, dtype=np.float16)
        fact.forward_mindspore_impl()
        _pynative_executor.sync()

    assert "return self.op(x)" in str(error_log.value)
    assert "tests/st/compiler/dfx/test_dfx_error_reporter.py" in str(
        error_log.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dfx_sort_exception_axis_value_in_bporp():
    """
    Feature: Dfx Check.
    Description: Check if the dfx meets the expected requirements.
    Expectation: Dfx meets the expected requirements.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    net = ReLUNetWithSortInBprop()
    grad_net = GradOfAllInputs(net, sens_param=False)
    grad_net.set_train()
    with pytest.raises(ValueError) as error_log:
        grad_net(input1)
        _pynative_executor.sync()

    assert "tests/st/compiler/dfx/test_dfx_error_reporter.py" in str(
        error_log.value)
    assert "return self.sort(x)" in str(error_log.value)
