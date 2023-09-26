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

import numpy as np
from tests.st.control.cases_register import case_register
import mindspore.context as context
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_cpu
@case_register.target_ascend
def test_dde_make_tuple_joined_with_tuple_output_primitive():
    """
    Feature: Eliminate unused element for tuple.
    Description: Two branch return make tuple and tuple output node like top_k
    Expectation: Correct result and no exception.
    """

    @ms.jit
    def topk_fun(x, k):
        if k == 0:
            output = (ms.ops.ones((0,), dtype=ms.float32), ms.ops.ones((0,), dtype=ms.int32))
        else:
            output = ms.ops.topk(x, k, None, True, True)
        return output

    x = ms.tensor([1., 2., 3.])
    k = ms.tensor([0])
    out = topk_fun(x, k)
    expect_out0 = ms.ops.ones((0,), dtype=ms.float32)
    expect_out1 = ms.ops.ones((0,), dtype=ms.int32)
    assert np.allclose(out[0].asnumpy(), expect_out0.asnumpy())
    assert np.allclose(out[1].asnumpy(), expect_out1.asnumpy())
