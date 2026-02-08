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
# ============================================================================
"""run os.env test"""
import pytest 
import os
import numpy as np
import mindspore as ms
from mindspore._c_expression import get_code_extra
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import match_array

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_os_env_mapping_get():
    """
    Feature: collections.abc.Mapping.get
    Description: get os.env key by collections.abc.Mapping.get
    Expectation: 0 break count
    """
    def func():
        device_id = os.environ.get("DEVICE_ID")
    os.environ["DEVICE_ID"] = "3"
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_os_env_mapping_get_with_set():
    """
    Feature: collections.abc.Mapping.get
    Description: get os.env key by collections.abc.Mapping.get
    Expectation: 0 break count
    """
    def func():
        device_id = os.environ.get("DEVICE_ID")
    os.environ["DEVICE_ID"] = "3"
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("a", [ms.Tensor(np.random.randn(2, 2).astype(np.float32))])
@pytest.mark.parametrize("b", [ms.Tensor(np.random.randn(2, 2).astype(np.float32))])
def test_os_env_mapping_get_with_tensor(a, b):
    """
    Feature: collections.abc.Mapping.get
    Description: get os.env key by collections.abc.Mapping.get
    Expectation: 0 break count
    """
    def func(a, b):
        if os.environ.get("DEVICE_ID"):
            return a * b
        return a + b
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ["DEVICE_ID"] = "3"
    jit(function=func, capture_mode="bytecode")(a, b)
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_os_env_branch_when_env_set():
    """
    Feature: @jit bytecode env branch.
    Description: Execute a bytecode-compiled cell that branches on an environment variable set to trigger multiplication.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_ai4sci.py::test_pijit_ai4sci_os_environ_set
    """
    class EnvBranchNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.flag = "ME_ENV"

        def construct(self, x):
            out = x
            if os.environ.get(self.flag) == '1':
                out = out * out
            else:
                out = out + x
            return out

    env_key = "ME_ENV"
    original_value = os.environ.get(env_key)
    x_np = np.array([1.0, 2.0], np.float32)

    try:
        os.environ[env_key] = '1'

        pynative_net = EnvBranchNet()
        pynative_result = pynative_net(Tensor(x_np))

        jit_net = EnvBranchNet()
        jit_net.construct = jit(jit_net.construct, capture_mode="bytecode")
        jit_result = jit_net(Tensor(x_np))
    finally:
        if original_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = original_value

    match_array(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_os_env_branch_when_env_unset():
    """
    Feature: @jit bytecode env branch.
    Description: Execute a bytecode-compiled cell when the environment variable is removed and addition path is taken.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_ai4sci.py::test_pijit_ai4sci_os_environ_unset
    """
    class EnvBranchNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.flag = "ME_ENV"

        def construct(self, x):
            out = x
            if os.environ.get(self.flag) == '1':
                out = out * out
            else:
                out = out + x
            return out

    env_key = "ME_ENV"
    original_value = os.environ.get(env_key)
    x_np = np.array([1.0, 2.0], np.float32)

    try:
        os.environ[env_key] = '1'
        os.environ.pop(env_key, None)

        pynative_net = EnvBranchNet()
        pynative_result = pynative_net(Tensor(x_np))

        jit_net = EnvBranchNet()
        jit_net.construct = jit(jit_net.construct, capture_mode="bytecode")
        jit_result = jit_net(Tensor(x_np))
    finally:
        if original_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = original_value

    match_array(pynative_result, jit_result)
