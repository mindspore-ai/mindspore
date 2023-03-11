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
""" test graph joinedstr """
import pytest
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_joinedstr_basic_variable_gpu():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x, y):
        if (x > 2 * y).all():
            res = f"res: {2 * y}"
        else:
            res = f"res: {x}"
        return res

    with pytest.raises(RuntimeError, match="Invalid value:res:"):
        input_x = Tensor(np.array([1, 2, 3, 4, 5]))
        out = joined_net(input_x, input_x)
        print("out:", out)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_joinedstr_basic_variable_ascend():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x, y):
        if (x > 2 * y).all():
            res = f"res: {2 * y}"
        else:
            res = f"res: {x}"
        return res

    with pytest.raises(RuntimeError) as v:
        input_x = Tensor(np.array([1, 2, 3, 4, 5]))
        out = joined_net(input_x, input_x)
        assert out == "x: [1, 2, 3, 4, 5]"
    assert "Illegal input dtype: String" in str(v.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_joinedstr_basic_variable_2():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x, y):
        if (x > 2 * y).all():
            res = f"{2 * y}"
        else:
            res = f"{x}"
        return res

    input_x = Tensor(np.array([1, 2, 3, 4, 5]))
    out = joined_net(input_x, input_x)
    assert str(out) == "(Tensor(shape=[5], dtype=Int64, value= [1, 2, 3, 4, 5]),)"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_joinedstr_out_tensor():
    """
    Feature: Support joinedstr.
    Description: Support joinedstr.
    Expectation: No exception.
    """
    @jit
    def joined_net(x):
        return f"x: {x}"

    input_x = Tensor([1, 2, 3])
    out = joined_net(input_x)
    assert str(out) == "('x: ', Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]))"
