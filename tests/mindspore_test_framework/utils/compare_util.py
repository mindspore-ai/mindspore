# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Utils for baseline comparison test."""

import numpy as np

from . import keyword
from .other_util import to_numpy_list


def compare(expect, func_result, baseline):
    """
    Compare results of function with baseline functions.

    Args:
        expect (dict): Config item in form of {'desc_expect': {'compare_with': f, 'max_error': 1e-3}}.
        func_result (dict): Verification item in form of {'result': Tensor([2, 2]), 'desc_inputs': Tensor([2, 2])}.
        baseline (str): Config item, compare_with | compare_gradient_with.
    Returns:
    """
    results = to_numpy_list(func_result[keyword.result])
    inputs = to_numpy_list(func_result[keyword.desc_inputs])
    funcs = expect[keyword.desc_expect][baseline]
    max_error = expect[keyword.desc_expect].get(keyword.max_error, 1e-3)
    for func in funcs:
        if isinstance(func, tuple):
            ret = func[0](func[1], *inputs)
        else:
            ret = func(*inputs)

        expects = to_numpy_list(ret)
        for i, e in enumerate(expects):
            if np.fabs(e - results[i]).max() > max_error:
                raise TypeError(f'Error: expect {e} by {func}, but got {results[i]}')
