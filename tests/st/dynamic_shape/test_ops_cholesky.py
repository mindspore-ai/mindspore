# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.st.utils import test_utils

from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import auto_generate as P
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def cholesky_forward_func(x, upper):
    return P.Cholesky(upper)(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cholesky_cpu():
    """
    Feature: Cholesky cpu kernel.
    Description: Test cholesky cpu kernel for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU", precompile_only=True)
    x = Tensor([[1.0, 1.0], [1.0, 2.0]], mstype.float32)
    output = cholesky_forward_func(x, True)
    assert output is None
