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

"""no_inline"""
from __future__ import absolute_import


def no_inline(fn=None):
    """
    Make the function to be reusable. The corresponding sub graph will not be inline.

    Args:
        fn (function): It is the python function. If it is a methon of a cell, please
            refer to :func:`mindspore.lazy_inline`.

    Returns:
        function, original function.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import no_inline, Tensor, jit
        >>> @no_inline
        ... def no_inline_fun(val):
        ...     x = val * 3 + 2
        ...     return x
        >>> @jit
        ... def call_no_inline_fun(val):
        ...     for _ in range(100):
        ...         val = no_inline_fun(val)
        ...     return val
        >>> call_no_inline_fun(Tensor(10))
    """

    def no_inline_wrap(fn):
        fn.no_inline = True
        return fn

    if fn is not None:
        return no_inline_wrap(fn)
    return no_inline_wrap
