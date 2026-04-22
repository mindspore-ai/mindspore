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
"""Test PIJit call mint functional primitive"""
import numpy as np
import mindspore as ms
from mindspore.mint import clamp
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_executed_by_graph_mode
from tests.st.pi_jit.share.utils import pi_jit_with_config

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_mint_clamp_with_tensor():
    """
    Feature: Functional.
    Description: Test functional feature with clamp.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x, min, max):  # pylint: disable=redefined-builtin
        return clamp(x, min, max)

    x = ms.Tensor([1, 2, 3, 4, 5])
    out_clamp_tensor = func(x, ms.Tensor(2), ms.Tensor(4))
    expect = ms.Tensor([2, 2, 3, 4, 4])
    assert np.all(out_clamp_tensor.asnumpy() == expect.asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_mint_clamp_with_scalar():
    """
    Feature: Functional.
    Description: Test functional feature with clamp.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x, min, max):  # pylint: disable=redefined-builtin
        return clamp(x, min, max)

    x = ms.Tensor([1, 2, 3, 4, 5])
    out_clamp_scalar = func(x, 2, 4)
    expect = ms.Tensor([2, 2, 3, 4, 4])
    assert np.all(out_clamp_scalar.asnumpy() == expect.asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_mint_clamp_default():
    """
    Feature: Functional.
    Description: Test functional feature with clamp.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x, min):  # pylint: disable=redefined-builtin
        return clamp(x, min)

    out = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    expect = ms.Tensor([2, 2, 3, 4, 5])
    assert np.all(out.asnumpy() == expect.asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_mint_clamp_keyword():
    """
    Feature: Functional.
    Description: Test functional feature with clamp.
    Expectation: Run success
    """
    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def func(x, min):  # pylint: disable=redefined-builtin
        return clamp(x, min, max=ms.Tensor(4)), clamp(x, max=3, min=1)

    out_clamp_tensor, out_clamp_scalar = func(ms.Tensor([1, 2, 3, 4, 5]), ms.Tensor(2))
    assert np.all(out_clamp_tensor.asnumpy() == ms.Tensor([2, 2, 3, 4, 4]).asnumpy())
    assert np.all(out_clamp_scalar.asnumpy() == ms.Tensor([1, 2, 3, 3, 3]).asnumpy())
    assert_executed_by_graph_mode(func)
