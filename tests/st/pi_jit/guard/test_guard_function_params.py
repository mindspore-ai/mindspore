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
"""Test guard for function params"""

from mindspore import Tensor, jit, nn, ops

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import match_array, assert_executed_by_graph_mode
from tests.st.pi_jit.conftest import run_in_subprocess


@run_in_subprocess({'GLOG_v': '1', 'MS_SUBMODULE_LOG_v': '{PI:0}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_for_function_param_of_user_defined_object():
    """
    Feature: Guard behavior for a function parameter of a user-defined object.
    Description:
        The function takes an `InferenceParams` object as one of its inputs. Each call passes a new instance:
        1) The first two calls use different `value` fields, which should trigger a recompile on the second call.
        2) The third call uses the same `value` as the second call, which should NOT trigger a recompile.
    Expectation: Numerical results are correct and there is no graph break.
    """

    class InferenceParams:
        def __init__(self, value: int):
            self.value = value

    class Net(nn.Cell):
        def construct(self, x: Tensor, params: InferenceParams):
            return x + params.value

    pynative_net = Net()
    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode', fullgraph=True)

    x = ops.randn(2, 4)

    # 1st call: value = 1, expect call_count = 1 (for graph-1)
    p1 = InferenceParams(1)
    ref1 = pynative_net(x, p1)
    out1 = jit_net(x, p1)
    match_array(out1, ref1)
    assert_executed_by_graph_mode(jit_net.construct, call_count=1)

    # 2nd call: value = 2 (different), expect recompile -> call_count = 1 (for graph-2)
    p2 = InferenceParams(2)
    ref2 = pynative_net(x, p2)
    out2 = jit_net(x, p2)
    match_array(out2, ref2)
    assert_executed_by_graph_mode(jit_net.construct, call_count=1)

    # 3rd call: value = 2 (same as previous), expect NO recompile -> call_count = 2 (for graph-2)
    p3 = InferenceParams(2)
    ref3 = pynative_net(x, p3)
    out3 = jit_net(x, p3)
    match_array(out3, ref3)
    assert_executed_by_graph_mode(jit_net.construct, call_count=2)
