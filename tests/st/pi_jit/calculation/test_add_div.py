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
"""Test add_div operation"""
import pytest 
from mindspore import dtype as mstype
from mindspore import ops, Tensor, jit
from ..share.utils import match_array
from tests.mark_utils import arg_mark
from mindspore._c_expression import get_code_extra

@jit(capture_mode="bytecode")
def jit_add_div(a, b, c):
    return a + b / c


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('ms_func', [jit_add_div])
@pytest.mark.parametrize('a', [1, 2.0, (1.0, 2.0, 3.0), [1.0, 2.0, 3.0], Tensor(ops.fill(mstype.float32, (2, 3), 8))])
@pytest.mark.parametrize('b', [3, 4.0, (4.0, 5.0, 6.0), [4.0, 5.0, 6.0], Tensor(ops.fill(mstype.float32, (2, 3), 8))])
@pytest.mark.parametrize('c', [5, 6.0, Tensor(ops.fill(mstype.float32, (2, 3), 8))])
def test_add_div(ms_func, a, b, c):
    """
    Feature: ALL TO ALL
    Description: test cases for add_div in PYNATIVE mode
    Expectation: the result match
    """
    if ((isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(c, (int, float))) or
       (isinstance(a, (float, Tensor)) and isinstance(b, (float, Tensor)) and isinstance(c, (float, Tensor))) or
       (isinstance(a, (tuple, Tensor)) and isinstance(b, (tuple, Tensor)) and isinstance(c, (tuple, Tensor))) or
       (isinstance(a, (list, Tensor)) and isinstance(b, (list, Tensor)) and isinstance(c, (list, Tensor)))):
       
       if isinstance(b, tuple) or isinstance(c, tuple):
        pytest.skip("b and c cannot be of type tuple.")
       if isinstance(b, list) or isinstance(c, list):
        pytest.skip("b and c cannot be of type list.")

       ms_res = ms_func(a, b, c)
       res = ms_func.__wrapped__(a, b, c)
       match_array(ms_res, res, error=1e-5, err_msg=str(ms_res))
       jcr = get_code_extra(ms_func.__wrapped__)
       assert(jcr['break_count_'] == 0)