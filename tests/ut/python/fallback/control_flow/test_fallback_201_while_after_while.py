# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test graph fallback control flow."""
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


def test_while_after_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_while():
        x = [1, 2, 3, 4]
        y = Tensor([8])
        z = 2
        while Tensor([sum(x)]) > y:
            x.append(z)
            y = Tensor([18])
        while y >= 0:
            y -= Tensor([x[0]])
        return Tensor(x), y
    res_x, res_y = control_flow_while_after_while()
    assert (res_x.asnumpy() == [1, 2, 3, 4, 2]).all()
    assert res_y == -1
