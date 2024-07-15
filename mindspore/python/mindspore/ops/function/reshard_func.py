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
    Specify the tensor by the given layout. The given layout must be type mindspore.Layout,
    can check :class:`mindspore.Layout` for reference.

    - In the Graph mode, this function can set the sharding propagation strategy of a tensor.
      For those tensor do not manually be set, their strategies are decided by the sharding
      strategy propagation algorithm automatically.
    - In the PyNative mode, this function can set a tensor sharding strategy in a Cell that
      runs in the Graph mode (i.e. inside the Cell processed by Cell.shard/F.shard).

    Note:
        - In the auto parallel mode, an exception will throw if the search mode is not
          "sharding_propagation".
        - In the semi-auto parallel mode, the parallel mode will automatically switch to auto
          parallel mode with the search mode be set to "sharding_propagation".

    Args:
        tensor (Tensor): The tensor to be set the sharding strategy.
        layout (Layout): The layout to shard the tensor precisely, including the device
                         arrangement (device_matrix) and the alias for the device matrix
                         (alias_name).

    Returns:
        Tensor. The mathematically equivalent of the input tensor.

    Raises:
        TypeError: Reshard takes in Tensor type as the first input param, but got: `type(tensor)`.
        TypeError: Reshard only support type mindspore.Layout but got: `type(layout)`.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import ops, nn, Tensor, context, Layout
        >>> context.set_context(mode=ms.GRAPH_MODE)
        >>> context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL,
        ...                                   search_mode="sharding_propagation")
        >>> class Network(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.matmul = ops.MatMul()
        ...         self.relu = ops.ReLU()
        ...     def construct(self, x, layout):
        ...         x = self.relu(x)
        ...         x_reshard = ops.reshard(x, self.layout)
        ...         y = Tensor(np.ones(shape=(128, 128)), dtype=ms.float32)
        ...         x = self.matmul(x_reshard, y)
        ...         return x
        >>>
        >>> layout = Layout((4, 2), ("dp", "mp"))
        >>> input_layout = layout("dp", "mp")
        >>> net = Network()
        >>> tensor = Tensor(np.ones(shape=(128, 128)), dtype=ms.float32)
        >>> out = net(tensor, input_layout)
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Reshard takes in Tensor type as the first input param, but got: {type(tensor)}.")
    if not isinstance(layout, Layout):
        raise TypeError(f"Reshard only support type mindspore.Layout, but got: {type(layout)}.")

    def layout_to_tuple(layout):
        layout_dict = layout.to_dict()
        tensor_map = layout_dict["tensor_map"]
        device_matrix_rev = layout_dict["device_matrix"][::-1]
        axis_stgy = ()
        for ind in tensor_map:
            if ind == -1:
                axis_stgy += (1,)
            else:
                axis_stgy += (device_matrix_rev[ind],)
        return axis_stgy

    in_strategy = layout_to_tuple(layout)
    _reshard = _get_cache_prim(P.Reshard)(in_layout=(layout,), out_layout=(layout,), in_strategy=(in_strategy,))
    return _reshard(tensor)

__all__ = [
    'reshard'
]

__all__.sort()
