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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn


@test_utils.run_with_cell
def format_cast_forward_func(x, acl_format):
    return ops.auto_generate.format_cast(x, acl_format)


@test_utils.run_with_cell
def format_cast_backward_func(x, acl_format):
    return ms.grad(format_cast_forward_func, (0,))(x, acl_format)


def set_context_mode(mode):
    if mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    elif mode == "ge":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize("mode", ["pynative", "kbk", "ge"])
def test_ops_format_cast(mode):
    """
    Feature: test ops.
    Description: test op format_cast.
    Expectation: success.
    """
    set_context_mode(mode)

    x_np = generate_numpy_ndarray_by_randn((2, 3, 4, 5), np.float16, "x")
    x = Tensor(x_np)
    acl_format = 29

    out = format_cast_forward_func(x, acl_format)
    np.testing.assert_allclose(out.asnumpy(), x_np, rtol=0)

    grad_out = format_cast_backward_func(x, acl_format)
    np.testing.assert_allclose(grad_out.asnumpy(), np.ones_like(x_np), rtol=0)
