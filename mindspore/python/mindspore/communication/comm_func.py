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

"""
Defines communication operators with functional form.
"""
from mindspore.communication import GlobalComm, get_group_rank_from_world_rank, get_group_size
from mindspore.communication.management import _get_group
# Internal re-export consumed by mint.distributed; keep the original helper identity.
from mindspore.communication._comm_helper import (  # pylint: disable=unused-import
    _get_group_rank_from_world_rank_from_cache_helper,
)
from mindspore.common.tensor import Tensor
from mindspore.common._decorator import deprecated
from mindspore._c_expression import TensorPy as Tensor_
from mindspore.ops import ReduceOp
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.primitive import _primexpr
from mindspore.ops.auto_generate.gen_ops_prim import inner_comm_all_reduce_op, inner_comm_all_gather_op
from mindspore._c_expression import CommHandle as CommHandle_
from mindspore import jit_class
import mindspore as ms

__all__ = [
    'all_reduce',
    'all_gather_into_tensor',
    'barrier',
    'broadcast',
    'gather_into_tensor',
    'reduce',
    'scatter_tensor',
]

import mindspore.ops.operations as P

_GROPU_SIZE_CACHE = {}

@jit_class
class CommHandle(CommHandle_):
    r"""
    Usually, handles are created in C++during the execution of communication operators and returned to the Python
    layer. It will not be created directly in Python. Only in scenarios where graph patterns are compatible,
    handles will be created using Python.
    """

    def __init__(self, handle=None, exec_sync=False):
        super(CommHandle, self).__init__() # pylint: disable=super-with-arguments
        self.handle = handle
        self.exec_sync = exec_sync


    def wait(self):
        r"""
        The wait for asynchronous handles will not take effect for handles created on the Python side.

        >>> import numpy as np
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import all_reduce
        >>> from mindspore import Tensor
        >>>
        >>> init()
        >>> input_tensor = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> output, handle = all_reduce(input_tensor, async_op=True)
        >>> handle.wait()
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]
        """
        if self.handle:
            self.handle.wait()
        if self.exec_sync:
            ms.runtime.synchronize()


    def __repr__(self):
        """Return string representation for debugging."""
        if self.handle:
            try:
                # Get the underlying CommHandle pointer address (this in C++)
                comm_handle_ptr_str = self.handle.get_comm_handle_ptr_str()
                return f"CommHandle(handle={self.handle}, CommHandle_ptr={comm_handle_ptr_str})"
            except Exception:  # pylint: disable=broad-exception-caught
                # __repr__ must never throw exceptions, so catch everything
                return f"CommHandle(handle={self.handle})"
        return "CommHandle(handle=None)"


default_handle = CommHandle()


@_primexpr
def _check_all_tensors(tensor_list):
    """check all elements in tensor_list are type of Tensor"""
    if not isinstance(tensor_list, (list, tuple)):
        raise TypeError(f"Expected list or tuple, but got {type(tensor_list)}.")
    for t in tensor_list:
        if not isinstance(t, Tensor):
            raise TypeError(f"Expected tensor, but got {type(t)}")


@_primexpr
def _check_all_tensor_same_dtype(*tensor_lists):
    """check all the input tensor has same dtype"""
    consistent_dtype = None
    for list_ in tensor_lists:
        if not isinstance(list_, (list, tuple)):
            list_ = [list_]
        for tensor_ in list_:
            if not isinstance(tensor_, Tensor):
                continue

            dtype = tensor_.dtype
            if consistent_dtype is None:
                consistent_dtype = dtype
            else:
                if dtype != consistent_dtype:
                    raise TypeError("all_to_all input dtype must be the same, "
                                    f"but got {consistent_dtype} and {dtype}.")


def _get_size(shape):
    numel = 1
    for s in shape:
        numel *= s
    return numel


def _is_split_sizes_empty(split_sizes):
    return split_sizes is None or not split_sizes


def _contiguous(tensor):
    if not tensor.is_contiguous() or tensor.storage_offset() != 0:
        tensor = tensor.contiguous()
    return tensor


@deprecated("2.9.0", "mindspore.ops.communication.all_reduce", False)
def all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP, async_op=False):
    """
    `mindspore.communication.comm_func.all_reduce` is deprecated from version 2.9.0 and will be
    removed in a future version. Please use :func:`mindspore.ops.communication.all_reduce` instead.

    Reduce tensors across all devices in such a way that all devices will get the same final result,
    returns the tensor which is all reduced.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        tensor (Tensor): The input tensor to be all reduced. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        op (str, optional): Specifies an operation used for element-wise reductions, like sum, prod, max, and min.
                  On the CPU, only 'sum' is supported. Default: ``ReduceOp.SUM`` .
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
                  means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.
        async_op (bool, optional): Whether this operator should be an async operator. Default: ``False`` .

    Returns:
        Tuple(Tensor, CommHandle), the output tensor has the same shape of the input,
        i.e., :math:`(x_1, x_2, ..., x_R)`. The contents depend on the specified operation.
        CommHandle is an async work handle, if `async_op` is set to True. CommHandle will be None,
        when `async_op` is False.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, or any of `op` and `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        Deprecated

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication as comm
        >>>
        >>> comm.init()
        >>> input_tensor = ms.Tensor(np.ones([2, 8]).astype(np.float32))
        >>> output, _ = comm.comm_func.all_reduce(input_tensor)
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]

    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For all_reduce, the input tensor must be tensor")
    if not isinstance(op, str):
        raise TypeError("For all_reduce, the input op type must be str")
    if op not in ('sum', 'prod', 'min', 'max'):
        raise TypeError("For all_reduce, the input op value must be one of sum, prod, min, max")
    group = _get_group(group)
    tensor = _contiguous(tensor)
    output = inner_comm_all_reduce_op(tensor, op, group, async_op)
    return _deal_dist_comm_outputs(output, async_op)


@deprecated("2.9.0", "mindspore.ops.communication.all_gather_into_tensor", False)
def all_gather_into_tensor(tensor, group=GlobalComm.WORLD_COMM_GROUP, async_op=False):
    """
    `mindspore.communication.comm_func.comm_func.comm_func.all_gather_into_tensor` is deprecated from version 2.9.0
    and will be removed in a future version. Please use :func:`mindspore.ops.communication.all_gather_into_tensor`
    instead.

    Gathers tensors from the specified communication group and returns the tensor which is all gathered.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        tensor (Tensor): The input tensor to be all gathered into tensor.
                        The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
            means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.
        async_op (bool, optional): Whether this operator should be an async operator. Default: ``False`` .

    Returns:
        Tuple(Tensor, CommHandle), if the number of devices in the group is N,
        then the shape of output tensor is :math:`(N, x_1, x_2, ..., x_R)`.
        CommHandle is an async work handle, if `async_op` is set to True.
        CommHandle will be None, when `async_op` is False.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, or `group` is not a str.
        ValueError: If the local rank id of the calling process in the group
                    is larger than the group's rank size.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        Deprecated

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication as comm
        >>>
        >>> comm.init()
        >>> input_tensor = ms.Tensor(np.ones([2, 8]).astype(np.float32))
        >>> output, _ = comm.comm_func.all_gather_into_tensor(input_tensor)
        >>> print(output)
        [[1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]]

    """

    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For all_gather_into_tensor, the input tensor must be tensor")
    group = _get_group(group)
    global _GROPU_SIZE_CACHE # pylint: disable=global-variable-not-assigned
    if group not in _GROPU_SIZE_CACHE:
        _GROPU_SIZE_CACHE[group] = get_group_size(group)
    group_size = _GROPU_SIZE_CACHE[group]
    tensor = _contiguous(tensor)
    output = inner_comm_all_gather_op(tensor, group_size, group, async_op)
    return _deal_dist_comm_outputs(output, async_op)


@deprecated("2.9.0", "mindspore.ops.communication.reduce", False)
def reduce(tensor, dst, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
    """
    `mindspore.communication.comm_func.reduce` is deprecated from version 2.9.0 and will be
    removed in a future version. Please use :func:`mindspore.ops.communication.reduce` instead.

    Reduces tensors across the processes in the specified communication group, sends the result
    to the target dst(global rank), and returns the tensor which is sent to the target process.

    Note:
        Only process with destination rank receives the reduced output.
        Only support PyNative mode, Graph mode is not currently supported.
        Other processes only get a tensor with shape [1], which has no mathematical meaning.

    Args:
        tensor (Tensor): The input tensor to be reduced. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dst (int): The target rank of the process(global rank) that receives the reduced output.
        op (str, optional): Specifies an operation used for element-wise reductions, like sum, prod, max, and min.
                On the CPU, only 'sum' is supported. Default: ``ReduceOp.SUM`` .
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
            means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        Tensor. Return the tensor in the specific rank of the process after reduction.
        The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, or any of `op` and `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        Deprecated

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.

            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication as comm
        >>>
        >>> # Launch 4 processes.
        >>> comm.init()
        >>> dest_rank=1
        >>> input_tensor = ms.Tensor(np.ones([2, 8]).astype(np.float32))
        >>> output = comm.comm_func.reduce(input_tensor, dst=dest_rank)
        >>> print(output)
        Process with rank 1: [[4. 4. 4. 4. 4. 4. 4. 4.]
                             [4. 4. 4. 4. 4. 4. 4. 4.]],
        Other processes: [0.].
    """

    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For reduce, the input tensor must be tensor")
    group_rank = get_group_rank_from_world_rank(dst, group)
    reduce_op = _get_cache_prim(P.Reduce)(dest_rank=group_rank, op=op, group=group)
    return reduce_op(tensor)


@deprecated("2.9.0", "mindspore.ops.communication.scatter_tensor", False)
def scatter_tensor(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP):
    r"""
    `mindspore.communication.comm_func.scatter_tensor` is deprecated from version 2.9.0 and will be
    removed in a future version. Please use :func:`mindspore.ops.communication.scatter_tensor` instead.

    Scatter tensor evenly across the processes in the specified communication group.

    Note:
        - The interface behavior only support Tensor input and scatter evenly, which
          is different from that of `pytorch.distributed.scatter`.
        - Only the tensor in process `src` (global rank) will do scatter.
        - Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The input tensor to be scattered. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        src (int, optional): Specifies the rank(global rank) of the process that send the tensor.
            And only process `src` will send the tensor.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``,
            which means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        Tensor, the shape of output is :math:`(x_1/src\_rank, x_2, ..., x_R)`. The dimension 0 of data is equal to
        the dimension of input tensor divided by `src`, and the other dimension keep the same.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, or any of `op` and `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        Deprecated

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication as comm
        >>>
        >>> # Launch 2 processes.
        >>>
        >>> comm.init()
        >>> input = ms.Tensor(np.arange(8).reshape([4, 2]).astype(np.float32))
        >>> out = comm.comm_func.scatter_tensor(tensor=input, src=0)
        >>> print(out)
        # rank_0
        [[0. 1.]
         [2. 3.]]
        # rank_1
        [[4. 5.]
         [6. 7.]]
    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For scatter_tensor, the input tensor must be tensor")
    if not isinstance(src, int):
        raise TypeError("For scatter_tensor, the src must be int")
    _src = get_group_rank_from_world_rank(src, group)
    _op = _get_cache_prim(P.CollectiveScatter)(_src, group)
    return _op(tensor)


@deprecated("2.9.0", "mindspore.ops.communication.gather_into_tensor", False)
def gather_into_tensor(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP):
    r"""
    `mindspore.communication.comm_func.gather_into_tensor` is deprecated from version 2.9.0 and will be
    removed in a future version. Please use :func:`mindspore.ops.communication.gather_into_tensor` instead.

    Gathers tensors from the specified communication group. The operation will gather the tensor
    from processes according to dimension 0.

    Note:
        Only the tensor in process `dst` (global rank) will keep the gathered tensor. The other process
        will keep a tensor with shape [1], which has no mathematical meaning.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The tensor to be gathered. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dst (int, optional): Specifies the rank(global rank) of the process that receive the tensor.
            And only process `dst` will receive the gathered tensor. Default: ``0``.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``,
            which means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        Tensor, the shape of output is :math:`(\sum x_1, x_2, ..., x_R)`. The dimension 0 of data is equal to
        sum of the dimension of input tensor, and the other dimension keep the same.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, `dst` is not an int,
            or `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        Deprecated

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication as comm
        >>>
        >>> # Launch 2 processes.
        >>>
        >>> comm.init()
        >>> input = ms.Tensor(np.arange(4).reshape([2, 2]).astype(np.float32))
        >>> output = comm.comm_func.gather_into_tensor(tensor=input, dst=0)
        >>> print(output)
        Process with rank 0: [[0. 1.],
                              [2. 3.],
                              [0. 1.],
                              [2. 3.]]
        Process with rank 1: [0]
    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For gather_into_tensor, the input tensor must be tensor")
    if not isinstance(dst, int):
        raise TypeError("For gather_into_tensor, the dst must be int")
    _dst = get_group_rank_from_world_rank(dst, group)
    _op = _get_cache_prim(P.CollectiveGather)(_dst, group)
    return _op(tensor)


@deprecated("2.9.0", "mindspore.ops.communication.broadcast", False)
def broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP):
    """
    `mindspore.communication.comm_func.broadcast` is deprecated from version 2.9.0 and will be
    removed in a future version. Please use :func:`mindspore.ops.communication.broadcast` instead.

    Broadcasts the tensor to the whole group.

    Note:
        - The tensors must have the same shape and format in all processes of the collection.
        - Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The tensor to be broadcasted. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        src (int, optional): Specifies the rank(global rank) of the process that broadcast the tensor.
            And only process `src` will broadcast the tensor.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``,
            which means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        Tensor, tensor has the same shape as input tensor :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `src` is not an integer or `group` is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        Deprecated

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication as comm
        >>>
        >>> # Launch 2 processes.
        >>>
        >>> comm.init()
        >>> data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
        >>> out = comm.comm_func.broadcast(tensor=data, src=0)
        >>> print(out)
        [[0. 1. 2. 3.]
         [4. 5. 6. 7.]]

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Broadcast
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#broadcast>`_

    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For broadcast, the input tensor must be tensor")
    if not isinstance(src, int):
        raise TypeError("For broadcast, the src must be int")
    _src = get_group_rank_from_world_rank(src, group)
    _op = _get_cache_prim(P.Broadcast)(_src, group)
    return _op((tensor,))[0]


@deprecated("2.9.0", "mindspore.ops.communication.barrier", False)
def barrier(group=GlobalComm.WORLD_COMM_GROUP):
    """
    `mindspore.communication.comm_func.barrier` is deprecated from version 2.9.0 and will be
    removed in a future version. Please use :func:`mindspore.ops.communication.barrier` instead.

    Synchronizes all processes in the specified group. Once the process call this operation, it will be blocked until
    all processes call this operation. After all processes finish calling the operations, the blocked processes
    will be woken and continue their task.

    Args:
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``,
            which means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Raises:
        RuntimeError: If backend is invalid, or distributed initialization fails.

    Supported Platforms:
        Deprecated

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import mindspore as ms
        >>> import mindspore.communication as comm
        >>>
        >>> # Launch 2 processes.
        >>> comm.init()
        >>> comm.comm_func.barrier()
        >>> print("barrier finish!")
        barrier finish!

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Barrier
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#barrier>`_
    """
    if not isinstance(group, str):
        raise TypeError(f"group must be type of string, but got {type(group)}")
    _op = _get_cache_prim(P.Barrier)(group)
    _op()
    ms.runtime.synchronize()


# Shared with mint.distributed send/recv and batched point-to-point operations.
def _deal_comm_outputs(output, async_op, exec_sync=False):
    """
    deal with comm ops outputs.
    """
    if isinstance(output, tuple):
        if not async_op:
            output[1].wait()
            if exec_sync:
                ms.runtime.synchronize()
            return (output[0], None)
        return (output[0], CommHandle(output[1], exec_sync))

    if not async_op:
        return (output, None)
    return (output, default_handle)


def _deal_dist_comm_outputs(output, async_op, exec_sync=False):
    """
    Deal with dist_comm outputs whose sync wait is handled inside the backend op.
    """
    if isinstance(output, tuple):
        if not async_op:
            if exec_sync:
                ms.runtime.synchronize()
            return (output[0], None)
        return (output[0], CommHandle(output[1], exec_sync))

    if not async_op:
        return (output, None)
    return (output, default_handle)
