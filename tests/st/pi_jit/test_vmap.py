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
''' test context option '''
import pytest
import numpy as np
import mindspore.ops.functional as F
from mindspore import dtype as mstype
from mindspore.common import Tensor
from mindspore.common.api import jit
from mindspore import context
from tests.mark_utils import arg_mark


@pytest.mark.skip(reason="pynative mode has an incorrect result")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_vmap_with_tuple_input():
    """
    Feature: vmap
    Description: When vmap use tuple inputs in graph, it must ensure the inputs is not eliminated.
    Expectation: success
    """
    def real_fn(x, y):
        return x * y

    def foo(fn):
        @jit(mode="PIJit")
        def wrapped(*args):
            def fn2(x, y):
                return F.jvp(fn, x, y)
            res = F.vmap(fn2)(args, args)
            return res
        return wrapped

    shape = (2, 3)
    context.set_context(mode=context.PYNATIVE_MODE)
    a = F.ones(shape, mstype.int32)
    b = F.ones(shape, mstype.int32) * 2
    res = foo(real_fn)(a, b)

    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], Tensor)
    assert isinstance(res[1], Tensor)
    assert np.allclose(res[0].asnumpy(), np.array([[2, 2, 2], [2, 2, 2]]))
    assert np.allclose(res[1].asnumpy(), np.array([[4, 4, 4], [4, 4, 4]]))
