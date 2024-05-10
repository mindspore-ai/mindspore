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

import mindspore.ops.operations as P
from mindspore.communication import GlobalComm, get_group_rank_from_world_rank
from mindspore.common.tensor import Tensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore.ops import ReduceOp
from mindspore.ops._primitive_cache import _get_cache_prim

__all__ = [
    'all_reduce',
    'all_gather_into_tensor',
    'reduce_scatter_tensor',
    'reduce',
    'P2POp',
    'batch_isend_irecv',
]


def comm_example():
    """
    Reduces the tensor data across all devices in such a way that all devices will get the same final result.
    """
    print("This is just an example of comm_func.py", P.AllReduce)


def all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
    """
    Reduce tensors across all devices in such a way that all deviceswill get the same final result,
    returns the tensor which is all reduced.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        tensor (Tensor): The input tensor to be all reduced. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        op (str, optional): Specifies an operation used for element-wise reductions, like sum, prod, max, and min.
                  On the CPU, only 'sum' is supported. Default: ``ReduceOp.SUM`` .
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
                  means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        Tensor, has the same shape of the input, i.e., :math:`(x_1, x_2, ..., x_R)`.
        The contents depend on the specified operation.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, or any of `op` and `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import all_reduce
        >>> from mindspore import Tensor
        >>>
        >>> init()
        >>> input_tensor = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> output = all_reduce(input_tensor)
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]

    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For all_reduce, the input tensor must be tensor")
    all_reduce_op = _get_cache_prim(P.AllReduce)(op=op, group=group)
    return all_reduce_op(tensor)


def all_gather_into_tensor(tensor, group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gathers tensors from the specified communication group and returns the tensor which is all gathered.

    Note:
        - The tensors must have the same shape and format in all processes of the collection.

    Args:
        tensor (Tensor): The input tensor to be all gathered into tensor.
                        The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
            means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        Tensor. If the number of devices in the group is N,
        then the shape of output is :math:`(N, x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, or `group` is not a str.
        ValueError: If the local rank id of the calling process in the group
                    is larger than the group's rank size.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

    >>> import numpy as np
    >>> import mindspore as ms
    >>> import mindspore.ops as ops
    >>> from mindspore.communication import init
    >>> from mindspore.communication.comm_func import all_gather_into_tensor
    >>> from mindspore import Tensor
    >>>
    >>> ms.set_context(mode=ms.GRAPH_MODE)
    >>> init()
    >>> input_tensor = Tensor(np.ones([2, 8]).astype(np.float32))
    >>> output = all_gather_into_tensor(input_tensor)
    >>> print(output)
    [[1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1.]]

    """

    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For all_gather_into_tensor, the input tensor must be tensor")
    all_gather_op = _get_cache_prim(P.AllGather)(group=group)
    return all_gather_op(tensor)


def reduce_scatter_tensor(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
    r"""
    Reduces and scatters tensors from the specified communication group and
    returns the tensor which is reduced and scattered.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        tensor(Tensor): The input tensor to be reduced and scattered, suppose it has a shape :math:`(N, *)`, where `*`
            means any number of additional dimensions. N must be divisible by rank_size.
            rank_size refers to the number of cards in the communication group.
        op (str, optional): Specifies an operation used for element-wise reductions,
                  like SUM and MAX. Default: ``ReduceOp.SUM`` .
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
            means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        Tensor, it has the same dtype as `input_x` with a shape of :math:`(N/rank\_size, *)`.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor, or any of `op` and `group` is not a str.
        ValueError: If the first dimension of the input cannot be divided by the rank_size.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import reduce_scatter_tensor
        >>> import numpy as np
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> input_tensor = Tensor(np.ones([8, 8]).astype(np.float32))
        >>> output = reduce_scatter_tensor(input_tensor)
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]

    """

    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For reduce_scatter_tensor, the input tensor must be tensor")
    reduce_scatter_op = _get_cache_prim(P.ReduceScatter)(op=op, group=group)
    return reduce_scatter_op(tensor)


def reduce(tensor, dst, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
    """
    Reduces tensors across the processes in the specified communication group, sends the result
    to the target dst(global rank), and returns the tensor which is sent to the target process.

    Note:
        Only process with destination rank receives the reduced output.
        Only support Pynative mode, Graph mode is not currently supported.
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
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.

            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> import mindspore.ops as ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import reduce
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> # Launch 4 processes.
        >>> init()
        >>> dest_rank=1
        >>> input_tensor = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> output = reduce(input_tensor)
        >>> print(output)
        Process with rank 1: [[4. 4. 4. 4. 4. 4. 4. 4.]
                             [4. 4. 4. 4. 4. 4. 4. 4.]],
        Other proesses: [0.].
    """

    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For reduce, the input tensor must be tensor")
    group_rank = get_group_rank_from_world_rank(dst, group)
    reduce_op = _get_cache_prim(P.Reduce)(dest_rank=group_rank, op=op, group=group)
    return reduce_op(tensor)


class P2POp:
    """
    Object for ``batch_isend_irecv``, to store information of ``"isend"`` and ``"irecv"``.

    Note:
        - Allow pass-in recv shape rather than tensor when ``op`` is ``"irecv"``.
        - ``tensor`` will not be modified in-place.

    Args:
        op(Union[str, function]: Only string of ``"isend"`` and ``"irecv"`` are allow.
                                 Or function of ``comm_func.isend`` and ``comm_func.irecv`` are allow.
        tensor(Union[Tensor, Tuple(int)]): tensor for sending/receiving or receive tensor shape
                                           when op is ``"irecv"``.
        peer(int): remote global rank for send/receive.
        tag(int): currently not supported yet. default: 0.
        recv_dtype(mindspore.dtype): when ``tensor`` is a tuple shape, this arg will be used and has
                                     to be configured. default: None

    Returns:
        P2POP Object.

    Raises:
        ValueError: when ``op`` is not string or function of ``"isend"`` and ``"irecv"``.
        TypeError: when ``tensor`` is not type of mindspore.Tensor or Tuple.
        NotImplementedError: when ``tag`` is not 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore.communication.comm_func import batch_isend_irecv, P2POp, isend, irecv
        >>> from mindspore import Tensor
        >>> send_tensor = Tensor(1.)
        >>> send_op = P2POp('isend', send_tensor, 1)
        >>> send_op = P2POp(isend, send_tensor, 1)
        >>> recv_tensor = Tensor(0.)
        >>> recv_op = P2POp('irecv', recv_tensor, 0)
        >>> recv_op = P2POp(irecv, recv_tensor, 0)
        >>> recv_op = P2POp('irecv', (), 0, recv_dtype=mindspore.float32)
    """
    def __init__(self, op, tensor, peer, group=None, tag=0, *, recv_dtype=None):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag
        self.recv_dtype = recv_dtype

    def __new__(cls, op, tensor, peer, group=None, tag=0, recv_dtype=None):
        if isinstance(op, str):
            op_name = op
        else:
            op_name = op.__name__
        if op_name not in ['isend', 'irecv']:
            raise ValueError(f"Expected ``op`` to be of type ``isend`` or `irecv``, but got {op_name}")
        if not isinstance(tensor, (Tensor, tuple)):
            raise TypeError(f"Expected ``tensor`` to be type of tuple of Tensor, but got {type(tensor)}.")
        if tag != 0:
            raise NotImplementedError("``tag`` not support yet.")
        return object.__new__(cls)


def batch_isend_irecv(p2p_op_list):
    """
    Batch send and recv tensors asynchronously.

    Note:
        - The ``isend`` and ``irecv`` of ``P2POp`` in ``p2p_op_list`` between ranks need to match each other.
        - ``P2POp`` in ``p2p_op_list`` can only use the same communication group.
        - ``tag`` of ``P2POp`` in ``p2p_op_list`` is not support yet.
        - Only support pynative mode, graph mode is not currently supported.

    Args:
        p2p_op_list(P2POp): list contains P2POps.

    Returns:
        tuple(Tensor). Output tensors is corresponding to ``p2p_op_list``:
        At P2POp with "isend" position, output tensor is a fake tensor with scalar, which has no meaning.
        At P2POp with "irecv" position, output tensor is a tensor received from remote end.

    Raises:
        TypeError: If ``p2p_op_list`` is not type of ``P2POp``.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore.communication import init, get_rank, get_group_size
        >>> from mindspore.communication.comm_func import batch_isend_irecv, P2POp
        >>> from mindspore import Tensor
        >>>
        >>> init()
        >>> this_rank = get_rank()
        >>> world_size = get_group_size()
        >>> next_rank = (this_rank + 1) % world_size
        >>> prev_rank = (this_rank + world_size - 1) % world_size
        >>>
        >>> send_tensor = Tensor(this_rank + 1, dtype=mindspore.float32)
        >>> recv_tensor = Tensor(0., dtype=mindspore.float32)
        >>>
        >>> send_op = P2POp('isend', send_tensor, next_rank)
        >>> recv_op = P2POp('irecv', recv_tensor, prev_rank)
        >>>
        >>> p2p_op_list = [send_op, recv_op]
        >>> output = batch_isend_irecv(p2p_op_list)
        >>> print(output)
        rank 0:
        (Tensor(shape=[], dtype=Float32, value= 0), Tensor(shape=[], dtype=Float32, value= 2))
        rank 1:
        (Tensor(shape=[], dtype=Float32, value= 0), Tensor(shape=[], dtype=Float32, value= 1))
    """
    send_tensors = []
    op_types = []
    remotes_ranks = []
    receive_shapes = []
    receive_dtypes = []
    tags = []
    group = p2p_op_list[0].group
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    type_ = None
    for i, p2p_op in enumerate(p2p_op_list):
        if not isinstance(p2p_op, P2POp):
            raise TypeError("must be type of P2POp")
        if isinstance(p2p_op.op, str):
            type_ = p2p_op.op
        else:
            type_ = p2p_op.op.__name__
        rank_ = p2p_op.peer if p2p_op.group is None else \
            get_group_rank_from_world_rank(p2p_op.peer, p2p_op.group)
        remotes_ranks.append(rank_)
        tags.append(p2p_op.tag)
        if type_ == "isend":
            send_tensors.append(p2p_op.tensor)
        elif type_ == "irecv":
            if isinstance(p2p_op.tensor, Tensor):
                receive_shapes.append(p2p_op.tensor.shape)
                receive_dtypes.append(p2p_op.tensor.dtype)
            elif isinstance(p2p_op.tensor, tuple):
                receive_shapes.append(p2p_op.tensor)
                if p2p_op.recv_dtype is None:
                    raise ValueError(f"'recv_dtype' of {i}th P2POp in p2p_op_list is None but op_types is"
                                     "'irecv' and P2POp.tensor is a tuple type.")
                receive_dtypes.append(p2p_op.recv_dtype)
            else:
                raise TypeError("p2p_op.tensor must be tensor or shape")
        else:
            raise TypeError("p2p_op.op must be isend or irecv")
        op_types.append(type_)

    _op = _get_cache_prim(P.BatchISendIRecv)(op_types,
                                             remotes_ranks,
                                             receive_shapes,
                                             receive_dtypes,
                                             group)
    output = _op(send_tensors)
    return output
