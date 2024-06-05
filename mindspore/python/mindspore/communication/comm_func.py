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
from mindspore.communication import GlobalComm, get_group_rank_from_world_rank, get_group_size
from mindspore.common.tensor import Tensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore.ops import ReduceOp, cat
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.primitive import _primexpr

__all__ = [
    'all_reduce',
    'all_gather_into_tensor',
    'all_to_all_with_output_shape',
    'all_to_all_single_with_output_shape',
    'barrier',
    'broadcast',
    'gather_into_tensor',
    'isend',
    'irecv',
    'reduce_scatter_tensor',
    'reduce',
    'scatter_tensor',
    'P2POp',
    'batch_isend_irecv',
]


def _check_split_sizes_sequence(tensor, sequence):
    if sequence == []:
        raise TypeError(f"sequence can not be empty list.")
    element0 = sequence[0]
    for idx in range(1, len(sequence)):
        if sequence[idx] != element0:
            raise TypeError(f"sequence containing different elements is not supported yet. "
                            f"Elements must be the same.")
    if sum(sequence) != tensor.shape[0]:
        raise TypeError(f" The sum of sequence should equal to tensor.shape[0].")


def _check_compute_split_count(tensor, output_split_sizes, input_split_sizes, group):
    """
    Check the output_split_sizes and input_split_sizes by the rules in _check_split_sizes_sequence,
        compute the split count and return it.
    """
    group_size = get_group_size(group)
    if output_split_sizes:
        _check_split_sizes_sequence(tensor, output_split_sizes)
        output_split_value = output_split_sizes[0]
    else:
        output_split_value = None
    if input_split_sizes:
        _check_split_sizes_sequence(tensor, input_split_sizes)
        input_split_value = input_split_sizes[0]
    else:
        input_split_value = None
    split_count = 0
    if input_split_value and output_split_value is None:
        split_count = tensor.shape[0] // input_split_value
    elif input_split_value is None and output_split_value:
        split_count = tensor.shape[0] // output_split_value
    elif input_split_value and output_split_value:
        if input_split_value != output_split_value:
            raise TypeError(f"input_split_value should equal to output_split_value.")
        split_count = tensor.shape[0] // input_split_value
    else:
        split_count = group_size
    return split_count


@_primexpr
def _check_all_tensors(tensor_list):
    """check all elements in tensor_list are type of Tensor"""
    if not isinstance(tensor_list, (list, tuple)):
        raise TypeError(f"Expected list or tuple, but got {type(tensor_list)}.")
    for t in tensor_list:
        if not isinstance(t, Tensor):
            raise TypeError(f"Expected tensor, but got {type(t)}")


@_primexpr
def _check_all_tensors_or_tuple(tensor_list):
    """check all elements in tensor_list are type of Tensor or tuple or list"""
    if not isinstance(tensor_list, (list, tuple)):
        raise TypeError(f"Expected list or tuple, but got {type(tensor_list)}.")
    for t in tensor_list:
        if not isinstance(t, (Tensor, tuple, list)):
            raise TypeError(f"Expected tensor or tuple, but got {type(t)}")


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
        >>> from mindspore import ops
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
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.

            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> from mindspore import ops
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
            raise TypeError(f"Expected ``tensor`` to be type of tuple or Tensor, but got {type(tensor)}.")
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
        - Only support PyNative mode, Graph mode is not currently supported.

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


def scatter_tensor(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP):
    r"""
    Scatter tensor evently across the processes in the specified communication group.

    Note:
        The interface behavior only support Tensor input and scatter evenly, which
        is different from that of `pytoch.distributed.scatter`.
        Only the tensor in process `src` (global rank) will do scatter.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor):  The input tensor to be scattered. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        src (int, optional): Specifies the rank(global rank) of the process that send the tensor.
            And only process `src` will send the tensor.
        group (str, optional): The communication group to work on.
            Default: "GlobalComm.WORLD_COMM_GROUP".

    Returns:
        Tensor, the shape of output is :math:`(x_1/src\_rank, x_2, ..., x_R)`. The dimension 0 of data is equal to
        the dimension of input tensor divided by `src`, and the other dimension keep the same.

    Raise:
        TypeError: If the type of the first input parameter is not Tensor, or any of `op` and `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import scatter_tensor
        >>> import numpy as np
        >>> # Launch 2 processes.
        >>>
        >>> init()
        >>> input = ms.Tensor(np.arange(8).reshape([4, 2]).astype(np.float32))
        >>> out = scatter_tensor(tensor=data, src=0)
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


def gather_into_tensor(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP):
    r"""
    Gathers tensors from the specified communication group. The operation will gather the tensor
    from processes according to dimension 0.

    Note:
        Only the tensor in process `dst` (global rank) will keep the gathered tensor. The other process
        will keep a tensor with shape [1], which has no mathematical meaning.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The tensor to be gathered. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dst(int, optional): Specifies the rank(global rank) of the process that receive the tensor.
            And only process `dst` will receive the gathered tensor. Default: 0.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Returns:
        Tensor, the shape of output is :math:`(\sum x_1, x_2, ..., x_R)`. The dimension 0 of data is equal to
        sum of the dimension of input tensor, and the other dimension keep the same.

    Raise:
        TypeError: If the type of the first input parameter is not Tensor, or any of `op` and `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore.communication.comm_func import gather_into_tensor
        >>> # Launch 2 processes.
        >>>
        >>> init()
        >>> input = Tensor(np.arange(4).reshape([2, 2]).astype(np.float32))
        >>> output = gather_into_tensor(tensor=data, dst=0)
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


def broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP):
    """
    Broadcasts the tensor to the whole group.

    Note:
        The tensors must have the same shape and format in all processes of the collection.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The tensor to be broadcasted. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        src (int, optional): Specifies the rank(global rank) of the process that broadcast the tensor.
            And only process `src` will broadcast the tensor.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Returns:
        Tensor, tensor has the same shape as input tensor :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If src is not an integer or group is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import broadcast
        >>> import numpy as np
        >>> # Launch 2 processes.
        >>>
        >>> init()
        >>> data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
        >>> out = broadcast(tensor=data, src=0)
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


def barrier(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Synchronizes all processes in the specified group. Once the process call this operation, it will be blocked until
    all processes call this operation. After all processes finish calling the operations, the blocked processes
    will be woken and continue their task.

    Args:
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Raises:
        RuntimeError: If backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import barrier
        >>> # Launch 2 processes.
        >>> init()
        >>> barrier()

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Barrier
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#barrier>`_
    """
    _op = _get_cache_prim(P.Barrier)(group)
    return _op()


def isend(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0):
    """
    Send tensors to the specified dest_rank.

    Note:
        Send and Receive must be used in combination and have same tag.

    Args:
        tensor (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dst (int, optional): A required integer identifying the destination rank(global rank). Default: 0.
        group (str, optional): The communication group to work on.
            Default: "hccl_world_group" on Ascend, "nccl_world_group" on GPU.
        tag (int, optional): A required integer identifying the send/recv message tag. The message will
            be received by the Receive op with the same "tag". Default: 0.

    Raises:
        TypeError: `dst` is not an int or `group` is not a str。
        ValueError: If the rank ID of the process is greater than the rank size of the communication group.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> from mindspore import ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import isend
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>>
        >>> init()
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> isend(input_, 0)
    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For isend, the input tensor must be tensor")
    _dst = get_group_rank_from_world_rank(dst, group)
    _op = _get_cache_prim(P.Send)(tag, _dst, group, group)
    _depend = _get_cache_prim(P.Depend)()
    return _depend(tensor, _op(tensor))


def irecv(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0):
    """
    Receive tensors from src.

    Note:
        Send and Receive must be used in combination and have same tag.
        The shape and dtype of input `tensor` is used to receive tensor, but the value
        of input `tensor` would not take effect.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The shape and dtype of this
            tensor is used to receive tensor, but the value of input `tensor` would not take effect.
        src (int, optional): A required integer identifying the source rank(global rank). Default: 0.
        group (str, optional): The communication group to work on.
            Default: "hccl_world_group" on Ascend, "nccl_world_group" on GPU.
        tag (int, optional): A required integer identifying the send/recv message tag. The message will
            be received by the Send op with the same "tag". Default: 0.

    Returns:
        Tensor, the shape of output is :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `src` is not an int or `group` is not a str.
        ValueError: If the rank ID of the process is greater than the rank size of the communication group.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> from mindspore import ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import irecv
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>>
        # Launch 2 processes.
        Process 0 send the following array to Process 1
        [[ 0.  1.]
         [ 2.  3.]]
        >>> init()
        >>> x = ms.Tensor(np.zeros([2, 2]))
        # Process 1 receive tensor from Process 0.
        >>> out = irecv(x, src=0)
        >>> print(out)
        [[ 0.  1.]
         [ 2.  3.]]
    """
    _src = get_group_rank_from_world_rank(src, group)
    shape = tensor.shape
    dtype = tensor.dtype
    _op = _get_cache_prim(P.Receive)(tag, _src, shape, dtype, group, group)
    return _op(tensor)


def all_to_all_with_output_shape(output_shape_list, input_tensor_list, group=None):
    """
    scatter and gather list of tensor to/from all rank according to input/output tensor list.

    Note:
        tensor shape in 'output_shape_list' and 'input_tensor_list' should be match across ranks.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        output_shape_list (Union[Tuple(Tensor), List(Tensor), Tuple(Tuple(int))]): List of shape
            that indicate the gathered tensors shape from remote ranks.
        input_tensor_list (Union[Tuple(Tensor), List(Tensor)]):
            List of tensors to scatter to the remote rank.
        group (str, optional): The communication group to work on.
            Default: None, which means "hccl_world_group" on Ascend, "nccl_world_group" on GPU.

    Returns:
        Tuple(Tensor), the tensors gathered from remote ranks.

    Raises:
        TypeError: If 'input_tensor_list' is not list of tensors.
        TypeError: If 'output_shape_list' is not list of tuple or tensors.
        TypeError: If tensors in 'input_tensor_list' are not the same type.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore.communication import init, get_rank, get_group_size
        >>> from mindspore.communication.comm_func import all_to_all_with_output_shape
        >>> from mindspore import Tensor
        >>> from mindspore.ops import zeros
        >>>
        >>> init()
        >>> this_rank = get_rank()
        >>> if this_rank == 0:
        >>>     send_tensor_list = [Tensor(1.), Tensor([[2, 3], [4, 5.]])]
        >>>     recv_tensor_list = [(), (2,)]
        >>> if this_rank == 1:
        >>>     send_tensor_list = [Tensor([2, 2.]), Tensor([4, 5, 6, 7.])]
        >>>     recv_tensor_list = [(2, 2), (4,)]
        >>> output = all_to_all_with_output_shape(recv_tensor_list, send_tensor_list)
        >>> print(output)
        rank 0:
        (Tensor(shape=[], dtype=Float32, value= 1),
         Tensor(shape=[2], dtype=Float32, value= [2.00000000e+00, 2.00000000e+00]))
        rank 1:
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[2.00000000e+00, 3.00000000e+00],
         [4.00000000e+00, 5.00000000e+00]]),
         Tensor(shape=[4], dtype=Float32, value=[4.00000000e+00, 5.00000000e+00, 6.00000000e+00, 7.00000000e+00]))

    """

    _check_all_tensors(input_tensor_list)
    _check_all_tensors_or_tuple(output_shape_list)
    _check_all_tensor_same_dtype(input_tensor_list)
    send_numel_list = []
    send_flatten_tensor = []
    recv_numel_list = []
    recv_shape_list = []

    for tensor in input_tensor_list:
        send_numel_list.append(tensor.size)
        send_flatten_tensor.append(tensor.reshape(-1))
    for tensor in output_shape_list:
        if isinstance(tensor, Tensor):
            recv_numel_list.append(tensor.size)
            recv_shape_list.append(tensor.shape)
        else:
            _shape = tensor
            recv_numel_list.append(_get_size(_shape))
            recv_shape_list.append(_shape)

    _op = _get_cache_prim(P.AlltoAllV)(send_numel_list, recv_numel_list, group)
    send_flatten_tensor = cat(send_flatten_tensor)
    output = _op(send_flatten_tensor)
    result = []
    offset = 0
    for numel, shape in zip(recv_numel_list, recv_shape_list):
        result.append(output[offset:offset + numel].reshape(shape))
        offset = offset + numel
    return tuple(result)


def all_to_all_single_with_output_shape(output_shape, tensor, output_split_sizes=None,
                                        input_split_sizes=None, group=None):
    """
    scatter and gather input with split size to/from all rank, and return result in a single tensor.

    Note:
        'output_shape' and 'tensor' shape should be match across ranks.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        output_shape (Union(Tensor, Tuple(int))): shape to indicate the shape
          of tensor gathered concatenated from remote rank.
        tensor (Tensor): tensor to be scattered to remote rank.
        output_split_sizes (Union(Tuple(int), List(int))): output split size at dim 0. If set to None,
            it means equally split by ``world_size``. Default: None.
        input_split_sizes (Union(Tuple(int), List(int))): input split size at dim 0. If set to None,
            it means equally split by ``world_size``. Default: None.
        group (str, optional): The communication group to work on.
            Default: None, which means "hccl_world_group" on Ascend, "nccl_world_group" on GPU.

    Returns:
        Tensor, the tensors gathered concatenated from remote ranks.
        If the numel of tensor gathered from remote is zero, it will return a Tensor will value 0,
        which has no actual meanning.

    Raises:
        TypeError: If 'tensor' is not tensor.
        TypeError: If 'output_shape' is not tuple or tensors.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore.communication import init, get_rank, get_group_size
        >>> from mindspore.communication.comm_func import all_to_all_single_with_output_shape
        >>> from mindspore import Tensor
        >>> from mindspore.ops import zeros
        >>>
        >>> init()
        >>> this_rank = get_rank()
        >>> if this_rank == 0:
        >>>     output_shape = (3, 3)
        >>>     tensor = Tensor([[0, 1, 2.], [3, 4, 5], [6, 7, 8]])
        >>>     result = all_to_all_single_with_output_shape(output_shape, tensor, [2, 1], [2, 1])
        >>> if this_rank == 1:
        >>>     output_shape = (2, 3)
        >>>     tensor = Tensor([[9, 10., 11], [12, 13, 14]])
        >>>     result = all_to_all_single_with_output_shape(output_shape, tensor)
        >>> print(result)
        rank 0:
        [[ 0.  1.  2.]
         [ 3.  4.  5.]
         [ 9. 10. 11.]]
        rank 1:
        [[ 6.  7.  8.]
         [12. 13. 14.]]

    """

    _check_all_tensors([tensor])
    _check_all_tensors_or_tuple([output_shape])
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP

    if input_split_sizes is None or not input_split_sizes:
        _world_size = get_group_size(group)
        if tensor.shape[0] % _world_size != 0:
            raise ValueError("input shape at dim 0 must be divided by world_size, "
                             f"but got {tensor.shape[0]} and {_world_size}.")
        _split_size = tensor.shape[0] // _world_size
        input_split_sizes = (_split_size,) * _world_size
    if output_split_sizes is None or not output_split_sizes:
        _world_size = get_group_size(group)
        shape_dim_0 = None
        if isinstance(output_shape, Tensor):
            shape_dim_0 = output_shape.shape[0]
        else:
            shape_dim_0 = output_shape[0]
        if shape_dim_0 % _world_size != 0:
            raise ValueError("output shape at dim 0 must be divided by world_size, "
                             f"but got {shape_dim_0} and {_world_size}.")
        _split_size = shape_dim_0 // _world_size
        output_split_sizes = (_split_size,) * _world_size

    send_size_without_first_dim = _get_size(tensor.shape[1:])
    send_numel_list = [size * send_size_without_first_dim for size in input_split_sizes]

    recv_size_without_first_dim = None
    recv_shape_without_first_dim = None
    if isinstance(output_shape, Tensor):
        recv_shape_without_first_dim = output_shape.shape[1:]
        recv_size_without_first_dim = _get_size(recv_shape_without_first_dim)
    else:
        recv_shape_without_first_dim = output_shape[1:]
        recv_size_without_first_dim = _get_size(recv_shape_without_first_dim)
    recv_numel_list = [size * recv_size_without_first_dim for size in output_split_sizes]

    _op = _get_cache_prim(P.AlltoAllV)(send_numel_list, recv_numel_list, group)
    _input = tensor.reshape(-1)
    result = _op(_input)
    if any(recv_numel_list):
        result = result.reshape((-1,) + recv_shape_without_first_dim)

    return result
