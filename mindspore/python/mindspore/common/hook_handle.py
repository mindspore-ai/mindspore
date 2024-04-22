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
"""The removable handle for cell hook function."""
from __future__ import absolute_import
import weakref
from mindspore._c_expression import Tensor as Tensor_


class _TensorHookHandle:
    r"""
    A handle provides the ability to remote a tensor hook.

    Note:
        It is only supported in pynative mode and works when registering or removing hook function for tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        self.id = None

    def remove(self):
        """
        Remove the tensor hook function, which corresponds to this '_TensorHookHandle' object.

        Args:
            None.

        Returns:
            None.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def hook_fn(grad):
            ...     return grad * 2
            ...
            >>> def hook_test(x, y):
            ...     z = x * y
            ...     handle = z.register_hook(hook_fn)
            ...     z = z * y
            ...     handle.remove()
            ...     return z
            ...
            >>> ms_grad = ms.grad(hook_test, grad_position=(0,1))
            >>> output = ms_grad(Tensor(1, ms.float32), Tensor(2, ms.float32))
            >>> print(output)
            (Tensor(shape=[], dtype=Float32, value=4), Tensor(shape=[], dtype=Float32, value=4))
        """
        if self.id is not None:
            Tensor_.remove_hook(self.id)


class HookHandle:
    r"""
    It is the return object of forward pre hook function, forward hook function and backward hook function of Cell
    object. It corresponds to the cell hook function and is used to remove the cell hook function by calling 'remove()'.

    Note:
        It is only supported in pynative mode and works when registering or removing hook function for Cell object.

    Args:
        hook_cell (Cell): The Cell object with hook function registered on. Default value: None.
        hook_key (int): The key of cell hook function in dict. It is generated during cell hook function registration.
                        Default value: -1.
        hook_type (str): The type of cell hook function: '_forward_pre_hook', '_forward_hook' or '_cell_backward_hook'.
                         Default value: "".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self, hook_cell=None, hook_key=-1, hook_type=""):
        if hook_cell is not None:
            self._hook_cell = weakref.ref(hook_cell)
        else:
            self._hook_cell = hook_cell
        self._hook_key = hook_key
        self._hook_type = hook_type

    def __del__(self):
        self._hook_cell = None
        self._hook_key = None
        self._hook_type = None

    def remove(self):
        """
        Remove the cell hook function, which corresponds to this 'HookHandle' object.
        In order to prevent running failed when switching to graph mode, it is not recommended to call the `remove()`
        function in the construct function of Cell object.

        Args:
            None.

        Returns:
            None.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> import mindspore.nn as nn
            >>> from mindspore import Tensor
            >>> from mindspore.ops import GradOperation
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def forward_pre_hook_fn(cell_id, inputs):
            ...     print("forward inputs: ", inputs)
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.mul = nn.MatMul()
            ...         self.handle = self.mul.register_forward_pre_hook(forward_pre_hook_fn)
            ...
            ...     def construct(self, x, y):
            ...         x = x + x
            ...         x = self.mul(x, y)
            ...         return x
            >>> grad = GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
            forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1],
                            dtype=Float32, value= [ 1.00000000e+00]))
            >>> net.handle.remove()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32,
            value= [ 2.00000000e+00]))
        """
        if self._hook_cell is not None:
            hook_cell = self._hook_cell()
            if self._hook_type == "_forward_pre_hook" and self._hook_key in hook_cell._forward_pre_hook:
                del hook_cell._forward_pre_hook[self._hook_key]
            elif self._hook_type == "_forward_hook" and self._hook_key in hook_cell._forward_hook:
                del hook_cell._forward_hook[self._hook_key]
            elif self._hook_type == "_cell_backward_hook":
                hook_cell._cell_backward_hook.remove_backward_hook(self._hook_key)
