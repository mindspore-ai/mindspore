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
"""Defines parameter operators with functional form."""
from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.parallel.shard import Layout
from mindspore.common.tensor import Tensor

def reshard(tensor, layout):
    r"""
    Reshard tensor by the given layout.

    Args:
        tensor (Tensor): Any tensor instance in the network.
        layout (Layout): The layout to shard the tensor.

    Returns:
        Tensor. The mathematically equivalent of the input tensor.

    Raises:
        TypeError: Reshard takes in Tensor type as the first input param, but got: `type(tensor)`.
        TypeError: Reshard only support tuple of layout as input, where layout is type
                   mindspore.parallel.shard.Layout but got: `type(layout)`.
        TypeError: Reshard only support tuple of layout as input, where layout is type
                   mindspore.parallel.shard.Layout but got tuple of: `type(ele)`.

        Examples:
            >>> from mindspore.parallel.shard import Layout
            >>> class Network(nn.Cell):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.matmul = ops.MatMul()
            >>>         self.relu = ops.ReLU()
            >>>         layout = Layout((4, 2), ("dp", "mp"))
            >>>         self.layout = (layout("dp", "mp"),)
            >>>     def construct(self, x):
            >>>         x = self.matmul(x)
            >>>         x = ops.reshard(x, self.layout)
            >>>         x = ops.relu(x)
            >>>         return x
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Reshard takes in Tensor type as the first input param, but got: {type(tensor)}.")
    if not isinstance(layout, tuple):
        raise TypeError(f"Reshard only support tuple of layout as input, where layout is type "
                        f"mindspore.parallel.shard.Layout but got: {type(layout)}.")
    for ele in layout:
        if not isinstance(ele, Layout):
            raise TypeError(f"Reshard only support tuple of layout as input, where layout is type "
                            f"mindspore.parallel.shard.Layout but got tuple of: {type(ele)}.")

    _reshard = _get_cache_prim(P.Reshard)(in_layout=layout, out_layout=layout)
    return _reshard(tensor)

__all__ = [
    'reshard'
]

__all__.sort()
