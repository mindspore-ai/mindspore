# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Cache for Primitive."""
from __future__ import absolute_import

from mindspore.ops.primitive import constexpr
from mindspore.ops import Primitive

_PRIM_CACHE = {}


def _temp_func():
    return 0


@constexpr(check=False)
def _is_need_compile(func):
    # No matter what the value of mode is, in ms_function scenario, this function always returns true.
    return func is None


def _get_cache_prim(cls: Primitive) -> Primitive:
    """
    Wrapper function, get a primitive by it's all args.

    Args:
        cls (Primitive): The Primitive need be wrapped.

    Returns:
        Function, a new function with return a primitive by it's all args.

    Examples:
        >>> # Example1:
        >>> from mindspore.ops._primitive_cache import _get_cache_prim
        >>> input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
        >>> axis = [0, 1]
        >>> p=2
        >>> keep_dims=False
        >>> epsilon=1e-12
        >>> _lp_norm = _get_cache_prim(P.LpNorm)(axis, p, keep_dims, epsilon)
        >>> output = _lp_norm(input_x)
        >>> print(output)
        [ 9.165152 10.954452]
        >>> # Example2:
        >>> from mindspore.ops._primitive_cache import _get_cache_prim
        >>> input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
        >>> axis = [0, 1]
        >>> _lp_norm = _get_cache_prim(P.LpNorm)(axis, 2, keep_dims=False, epsilon=1e-12)
        >>> output = _lp_norm(input_x)
        >>> print(output)
        [ 9.165152 10.954452]
    """

    def _new_prim_for_graph(*args, **kwargs) -> Primitive:
        return cls(*args, **kwargs)

    def _get_cache_prim_for_pynative(*args, **kwargs) -> Primitive:
        """Get a primitive singleton by it's all args."""
        global _PRIM_CACHE
        key = (str(cls),)
        str_args = [str(arg) for arg in args]
        key += tuple(str_args)
        for attr_name in kwargs:
            attr_value = kwargs.get(attr_name)
            key += (attr_name + ":" + str(attr_value),)
        # Note: The key must be a str.
        key = str(key)
        if key not in _PRIM_CACHE:
            prim = Primitive.__new__(cls, *args, **kwargs)
            # Only init once.
            prim.__init__(*args, **kwargs)
            _PRIM_CACHE[key] = prim
        return _PRIM_CACHE.get(key)

    if _is_need_compile(_temp_func): # @jit.cond: True
        return _new_prim_for_graph
    return _get_cache_prim_for_pynative
