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
import mindspore as ms
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_bool_tensor_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([1, 2, 3]).asnumpy()
        return bool(all(x - [1, 2, 3]))

    out = foo()
    assert not out


@pytest.mark.skip(reason="RebuildKernelSelectBackoffOp Unsupported op[Shape].")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_bool_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test bool() in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return bool(x.asnumpy())

    x = Tensor([-1.0], ms.float32)
    res = foo(x)
    assert res
