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
""" test primitive with keyword arguments """
import numpy as np
import mindspore as ms
from mindspore import ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_primitive_with_part_keyword_by_search_sorted_forward():
    """
    Feature: Primitive
    Description: Test primitive with keyword arguments
    Expectation: No exception.
    """
    @ms.jit
    def search_sorted0(arg0, arg1, arg2, arg3, arg4):
        return ops.SearchSorted(arg0, arg1)(arg2, arg3, arg4)

    @ms.jit
    def search_sorted1(arg0, arg1, arg2, arg3, arg4):
        return ops.SearchSorted(arg0, right=arg1)(arg2, values=arg3, sorter=arg4)

    dtype = ms.int64
    right = False
    sorted_sequence = ms.Tensor(np.array([[0, 1, 3, 5, 7], [2, 4, 6, 8, 10]]), ms.float32)
    values = ms.Tensor(np.array([[3, 6, 9], [3, 6, 9]]), ms.float32)
    sorter = None

    output0 = search_sorted0(dtype, right, sorted_sequence, values, sorter)
    output1 = search_sorted1(dtype, right, sorted_sequence, values, sorter)
    assert np.allclose(output0.asnumpy(), output1.asnumpy(), 0.001, 0.001)
