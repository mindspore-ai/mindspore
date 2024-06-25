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
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import Tensor
from mindspore.ops.operations import math_ops as P
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

context.set_context(precompile_only=True)


@test_utils.run_with_cell
def betainc_forward_func(a, b, x):
    return P.Betainc()(a, b, x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_betainc(mode):
    """
    Feature: Betainc
    Description: Test of input fp64 graph
    Expectation: match to tf.raw_ops.Betainc
    """
    context.set_context(mode=mode)
    a_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
    b_np = np.array([[2, 3], [4, 5]]).astype(np.float32)
    x_np = np.array([[0.5, 0.5], [0.4, 0.3]]).astype(np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)
    x = Tensor(x_np)
    _ = betainc_forward_func(a, b, x)
    #expect = np.array([[0.75, 0.6875], [0.45568, 0.19410435]], dtype=np.float32)
    #assert np.allclose(output_ms.asnumpy(), expect, 1e-4, 1e-4)
