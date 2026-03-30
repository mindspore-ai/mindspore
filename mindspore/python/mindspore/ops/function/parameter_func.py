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

"""Defines parameter operators with functional form."""

from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import assign, assign_add, assign_sub


def index_add(x, indices, y, axis, use_lock=True, check_index_bound=True):
    """
    Add the elements of input `y` into input `x` along the given axis and indices.

    .. warning::
        After version 2.9.0, the parameter `x` will be renamed to `input`, and the parameter order
        and `alpha` semantics will change.

    .. note::
        - `indices` is a one-dimensional tensor, and :math:`indices.shape[0] = y.shape[axis]` .
        - The value range of the elements in `indices` is :math:`[0, x.shape[axis] - 1]` .

    Args:
        x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        y (Tensor): The input tensor to add to `x`.
        axis (int): The specified axis.
        use_lock (bool, optional): Whether to enable a lock to protect the updating process of variable tensors.
           Default ``True`` .
        check_index_bound (bool, optional): Whether to check index boundary. Default ``True`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.Parameter(mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.float32),
        ...                         name="name_x")
        >>> indices = mindspore.tensor([0, 2], mindspore.int32)
        >>> y = mindspore.tensor([[0.5, 1.0], [1.0, 1.5], [2.0, 2.5]], mindspore.float32)
        >>> output = mindspore.ops.index_add(x, indices, y, 1)
        >>> print(output)
        [[ 1.5  2.   4. ]
         [ 5.   5.   7.5]
         [ 9.   8.  11.5]]
    """
    _index_add = _get_cache_prim(P.IndexAdd)(axis, use_lock, check_index_bound)
    return _index_add(x, indices, y)


__all__ = [
    'assign',
    'assign_sub',
    'assign_add',
    'index_add'
]
__all__.sort()
