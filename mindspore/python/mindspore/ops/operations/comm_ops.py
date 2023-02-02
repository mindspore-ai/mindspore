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

"""Communication APIs.
"""
from __future__ import absolute_import
from __future__ import division

from mindspore.common import Tensor
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.communication.management import get_rank, get_group_size, GlobalComm, _get_group, _host_distribute
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import PrimitiveWithInfer, PrimitiveWithCheck, Primitive, prim_attr_register
from mindspore.common.api import context


class ReduceOp:
    """
    Operation options for reducing tensors. This is an enumerated type, not an operator.

    The main calling methods are as follows:

    - SUM: ReduceOp.SUM.
    - MAX: ReduceOp.MAX.
    - MIN: ReduceOp.MIN.
    - PROD: ReduceOp.PROD.

    There are four kinds of operation options, "SUM", "MAX", "MIN", and "PROD".

    - SUM: Take the sum.
    - MAX: Take the maximum.
    - MIN: Take the minimum.
    - PROD: Take the product.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with multiple devices.

        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor, ops, nn
        >>> from mindspore.ops import ReduceOp
        >>>
        >>> init()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.allreduce_sum = ops.AllReduce(ReduceOp.SUM)
        ...
        ...     def construct(self, x):
        ...         return self.allreduce_sum(x)
        ...
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_)
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]
    """
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    PROD = "prod"


def check_collective_target_dtype(data_name, data_dtype, prim_name):
    """Check if data type is valid."""
    default_target_dtypes = (mstype.int8, mstype.int32, mstype.float16, mstype.float32)
    gpu_target_dtypes = (mstype.bool_, mstype.int8, mstype.int32, mstype.int64, mstype.uint32, mstype.uint64,
                         mstype.float16, mstype.float32, mstype.float64)

    valid_dtype = gpu_target_dtypes if context.get_context("device_target") == "GPU" else default_target_dtypes
    validator.check_tensor_dtype_valid(data_name, data_dtype, valid_dtype, prim_name)


def check_hcom_group_valid(group, prim_name=None):
    """Check if hcom group is valid."""
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if not _host_distribute() and context.get_context("mode") == context.PYNATIVE_MODE and \
            group != GlobalComm.WORLD_COMM_GROUP:
        raise RuntimeError(f"{msg_prefix} 'group' only support 'hccl_world_group' in pynative mode, but got "
                           f"'group': {group}. Please start by using mpi-run.")


class AllReduce(Primitive):
    """
    Reduces the tensor data across all devices in such a way that all devices will get the same final result.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        op (str): Specifies an operation used for element-wise reductions, like sum, prod, max, and min.
                  On the CPU, only 'sum' is supported. Default: ReduceOp.SUM.
        group (str): The communication group to work on. Default: "GlobalComm.WORLD_COMM_GROUP".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape of the input, i.e., :math:`(x_1, x_2, ..., x_R)`.
        The contents depend on the specified operation.

    Raises:
        TypeError: If any of `op` and `group` is not a str,
                   or fusion is not an integer, or the input's dtype is bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with multiple devices.

        >>> import numpy as np
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore.ops import ReduceOp
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>>
        >>> init()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.allreduce_sum = ops.AllReduce(ReduceOp.SUM)
        ...
        ...     def construct(self, x):
        ...         return self.allreduce_sum(x)
        ...
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_)
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]
    """

    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize AllReduce."""
        if not isinstance(op, type(ReduceOp.SUM)):
            raise TypeError(f"For '{self.name}', the 'op' must be str, but got {type(op).__name__}.")
        if not isinstance(_get_group(group), str):
            raise TypeError(f"For '{self.name}', the 'group' must be str, "
                            f"but got {type(_get_group(group)).__name__}.")
        check_hcom_group_valid(group, prim_name=self.name)
        self.op = op
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 0)
        self.add_prim_attr('index', 0)
        self.add_prim_attr('no_eliminate', True)


class AllGather(PrimitiveWithInfer):
    """
    Gathers tensors from the specified communication group.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        group (str): The communication group to work on. Default: "GlobalComm.WORLD_COMM_GROUP".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor. If the number of devices in the group is N,
        then the shape of output is :math:`(N, x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `group` is not a str.
        ValueError: If the local rank id of the calling process in the group
                    is larger than the group's rank size.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.allgather = ops.AllGather()
        ...
        ...     def construct(self, x):
        ...         return self.allgather(x)
        ...
        >>> input_x = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_x)
        >>> print(output)
        [[1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]]
    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize AllGather."""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank = get_rank(_get_group(group))
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank, 'rank_size', self.rank_size, Rel.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 0)
        self.add_prim_attr('mean_flag', False)
        self.add_prim_attr('no_eliminate', True)

    def __call__(self, tensor):
        raise NotImplementedError

    def infer_shape(self, x_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.rank_size
        return x_shape

    def infer_dtype(self, x_dtype):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype


class _MiniStepAllGather(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do reducescatter in backward in mini-step. It is only for
    internal use of parallel modules and cannot be called by users.

    Args:
        group (str): The communication group to work on. Default: None.
        grad_accumulation_step (int): The grad accumulation step. Default: None.
    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP, grad_accumulation_step=None, mean_flag=None):
        """Initialize _MiniStepAllGather."""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank = get_rank(_get_group(group))
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank, 'rank_size', self.rank_size, Rel.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 1)
        self.grad_accumulation_step = grad_accumulation_step
        self.mean_flag = mean_flag
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape, z_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.rank_size
        return x_shape

    def infer_dtype(self, x_dtype, z_shape):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype


class _MicroStepAllGather(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do reducescatter in backward in mini-step. It is only for
    internal use of parallel modules and cannot be called by users.

    Args:
        group (str): The communication group to work on. Default: None.
    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP, mean_flag=None):
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank_size = 1
        if group != "":
            self.rank = get_rank(_get_group(group))
            self.rank_size = get_group_size(_get_group(group))
            validator.check('rank', self.rank, 'rank_size', self.rank_size, Rel.LT, self.name)
            self.add_prim_attr('rank_size', self.rank_size)
            self.add_prim_attr('group', _get_group(group))
            self.add_prim_attr('fusion', 1)
            self.add_prim_attr('do_mirror', False)
            self.mean_flag = mean_flag
            self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape, z_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.rank_size
        return x_shape

    def infer_dtype(self, x_dtype, z_dtype):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype


class _HostAllGather(PrimitiveWithInfer):
    """
    Gathers tensors from the specified communication group on host.

    Note:
        The tensors must have the same shape and format in all processes of the collection.
        _HostAllGather is a host-side operator, it depends on OpenMPI and must use build option -M on
        to enable it. Using mpirun command to run it:
        mpirun -output-filename log -merge-stderr-to-stdout -np 3 python test_host_all_gather.py

    Args:
        group (Union[tuple[int],list[int]]): The rand_ids of communication group to work on. Default: None.

    Raises:
        TypeError: If group is not a list nor tuple, or elements of group are not int.
        ValueError: If group is not set, or rank_id from group not in [0, 7].

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor. If the number of devices in the group is N,
        then the shape of output is :math:`(N, x_1, x_2, ..., x_R)`.
    """

    @prim_attr_register
    def __init__(self, group=None):
        """Initialize _HostAllGather."""
        if group is None:
            raise ValueError(f"For '{self.name}', the 'group' cannot be None, but got {group}.")
        validator.check_value_type('group', group, (tuple, list), self.name)
        validator.check_int(len(group), 2, Rel.GE, "group size", self.name)
        for r in group:
            validator.check_int_range(r, 0, 7, Rel.INC_BOTH, "rank_id", self.name)
            validator.check_value_type("rank_id", r, (int,), self.name)
        self.group_size = len(group)
        self.add_prim_attr('group', group)
        self.add_prim_attr('no_eliminate', True)
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.group_size
        return x_shape

    def infer_dtype(self, x_dtype):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype

    def __call__(self, tensor):
        raise NotImplementedError


class ReduceScatter(Primitive):
    r"""
    Reduces and scatters tensors from the specified communication group.
    For more details about it, please refer to `Distributed Set Communication Primitives - ReduceScatter \
    <https://www.mindspore.cn/tutorials/experts/en/master/parallel/communicate_ops.html#reducescatter>`_ .

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        op (str): Specifies an operation used for element-wise reductions,
                  like SUM and MAX. Default: ReduceOp.SUM.
        group (str): The communication group to work on. Default: "GlobalComm.WORLD_COMM_GROUP".

    Inputs:
        - **input_x** (Tensor) - Input Tensor, suppose it has a shape :math:`(N, *)`, where `*`
          means any number of additional dimensions. N must be divisible by rank_size.
          rank_size refers to the number of cards in the communication group.

    Outputs:
        Tensor, it has the same dtype as `input_x` with a shape of :math:`(N/rank\_size, *)`.

    Raises:
        TypeError: If any of operation and group is not a string.
        ValueError: If the first dimension of the input cannot be divided by the rank_size.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with 2 devices.

        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindspore.ops import ReduceOp
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.reducescatter = ops.ReduceScatter(ReduceOp.SUM)
        ...
        ...     def construct(self, x):
        ...         return self.reducescatter(x)
        ...
        >>> input_ = Tensor(np.ones([8, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_)
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]
    """

    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize ReduceScatter."""
        validator.check_value_type('op', op, (type(ReduceOp.SUM),), self.name)
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.op = op
        self.rank_size = get_group_size(_get_group(group))
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 0)
        self.add_prim_attr('no_eliminate', True)

    def __call__(self, tensor):
        raise NotImplementedError


class _HostReduceScatter(PrimitiveWithInfer):
    """
    Reduces and scatters tensors from the specified communication group on host.

    Note:
        The tensors must have the same shape and format in all processes of the collection.
        _HostReduceScatter is a host-side operator, it depends on OpenMPI and must use build option
        -M on to enable it. Using mpirun command to run it:
        mpirun -output-filename log -merge-stderr-to-stdout -np 3 python test_host_reduce_scatter.py

    Args:
        op (str): Specifies an operation used for element-wise reductions,
                  like sum, max, avg. Default: ReduceOp.SUM.
        group (Union[tuple[int],list[int]]): The rand_ids of communication group to work on. Default: None.

    Raises:
        TypeError: If op is not a string and group is not a list nor tuple,
                   or elements of group are not int.
        ValueError: If the first dimension of input can not be divided by group size,
                    or group is not set, or rank_id not in [0, 7].
    """

    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM, group=None):
        """Initialize _HostReduceScatter."""
        if group is None:
            raise ValueError(f"For '{self.name}', the 'group' cannot be None, but got {group}.")
        validator.check_value_type('op', op, (type(ReduceOp.SUM),), self.name)
        validator.check_value_type('group', group, (tuple, list), self.name)
        validator.check_int(len(group), 2, Rel.GE, "group size", self.name)
        for r in group:
            validator.check_int_range(r, 0, 7, Rel.INC_BOTH, "rank_id", self.name)
            validator.check_value_type("rank_id", r, (int,), self.name)
        self.op = op
        self.group_size = len(group)
        self.add_prim_attr('group', group)
        self.add_prim_attr('no_eliminate', True)
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape):
        if x_shape[0] % self.group_size != 0:
            raise ValueError(f"For '{self.name}', the first dimension of 'x_shape' must be divided by 'group_size', "
                             f"but got 'x_shape[0]': {x_shape[0]}, 'rank_size': {self.group_size}.")
        x_shape[0] = int(x_shape[0] / self.group_size)
        return x_shape

    def infer_dtype(self, x_dtype):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype

    def __call__(self, tensor):
        raise NotImplementedError


class Broadcast(PrimitiveWithInfer):
    """
    Broadcasts the tensor to the whole group.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        root_rank (int): Source rank. Required in all processes except the one
                   that is sending the data.
        group (str): The communication group to work on. Default: "GlobalComm.WORLD_COMM_GROUP".

    Inputs:
        - **input_x** (tuple[Tensor]) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape of the input, i.e., :math:`(x_1, x_2, ..., x_R)`.
        The contents depend on the data of the `root_rank` device.

    Raises:
        TypeError: If root_rank is not an integer or group is not a string.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with multiple devices.

        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.broadcast = ops.Broadcast(1)
        ...
        ...     def construct(self, x):
        ...         return self.broadcast((x,))
        ...
        >>> input_x = Tensor(np.ones([2, 4]).astype(np.int32))
        >>> net = Net()
        >>> output = net(input_x)
        >>> print(output)
        (Tensor(shape[2,4], dtype=Int32, value=
        [[1, 1, 1, 1],
         [1, 1, 1, 1]]),)
    """

    @prim_attr_register
    def __init__(self, root_rank, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize Broadcast."""
        validator.check_value_type('root_rank', root_rank, (int,), self.name)
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        check_hcom_group_valid(group, prim_name=self.name)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('no_eliminate', True)

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        if not isinstance(x_dtype, tuple):
            raise TypeError(f"For '{self.name}', the 'input_x' must be a tuple, but got {type(x_dtype).__name__}!")
        for _ele in x_dtype:
            check_collective_target_dtype('x', _ele, self.name)
        return x_dtype


class _AllSwap(PrimitiveWithCheck):
    """
    _AllSwap is a collective operation.

    _AllSwap sends data from the all processes to the all processes in the specified group. It has two phases:

    - The scatter phase: On each process, the operand is split into the send size of blocks along the
      0-th axis, and the blocks are scattered to all processes, e.g., the ith block is send to the ith process.
    - The gather phase: Each process concatenates the received blocks along the 0-th axis.

    Note:
        The tensors must have the same format in all processes of the collection.

    Args:
        group (str): The communication group name.

    Inputs:
        tensor_in (tensor): A 2-D tensor. On each process, divide blocks into number of the send size.
        send_size (tensor): A 1-D int64 tensor. The element is the send data size for each process.
        recv_size (tensor): A 1-D int64 tensor. The element is the receive data size for each process.

    Returns:
        tensor_out (tensor): The result tensor.

    Raises:
        TypeError: If group is not a string.
    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize _AllSwap"""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.init_prim_io_names(inputs=['tensor_in', 'send_size', 'recv_size'], outputs=['tensor_out'])
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('no_eliminate', True)
        self.add_prim_attr('order_enforce_skip', True)

    def __check__(self, tensor_in, send_size, recv_size):
        validator.check_subclass("tensor_in", tensor_in['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("send_size", send_size['dtype'], [mstype.int64],
                                           self.name)
        validator.check_tensor_dtype_valid("recv_size", recv_size['dtype'], [mstype.int64],
                                           self.name)

        validator.check_equal_int(len(tensor_in['shape']), 2, "tensor_in", self.name)
        validator.check_equal_int(len(send_size['shape']), 1, "send_size", self.name)
        validator.check_equal_int(len(recv_size['shape']), 1, "recv_size", self.name)

        out_shape = [-1] + [tensor_in['shape'][1]]
        out = {'shape': out_shape,
               'dtype': tensor_in['dtype'],
               'value': None}
        return out


class NeighborExchange(Primitive):
    """
    NeighborExchange is a collective operation.

    NeighborExchange sends data from the local rank to ranks in the send_rank_ids,
    as while receive data from recv_rank_ids.

    Note:
        The user needs to preset
        communication environment variables before running the following example, please check the details on the
        official website of `MindSpore \
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html#communication-operator>`_.

        This operator requires a full-mesh network topology, each device has the same vlan id, and the ip & mask are
        in the same subnet, please check the `details \
        <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#注意事项>`_.

    Args:
        send_rank_ids (list(int)): Ranks which the data is sent to.
        recv_rank_ids (list(int)): Ranks which the data is received from.
        recv_shapes (tuple(list(int))): Data shape which received from recv_rank_ids.
        send_shapes (tuple(list(int))): Data shape which send to the send_rank_ids.
        recv_type (type): Data type which received from recv_rank_ids
        group (str): The communication group to work on. Default: "GlobalComm.WORLD_COMM_GROUP".

    Inputs:
        - **input_x** (tuple[Tensor]) - Shapes are same as args of send_shapes.

    Outputs:
        Tuple tensor, shapes are same as args of recv_shapes.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> # This example should be run with 2 devices. Refer to the tutorial > Distributed Training on mindspore.cn
        >>> import os
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.neighborexchange = ops.NeighborExchange(send_rank_ids=[1], recv_rank_ids=[1],
        ...                                                      recv_shapes=([2, 2],), send_shapes=([3, 3],),
        ...                                                      recv_type=ms.float32)
        ...
        ...
        ...     def construct(self, x):
        ...         out = self.neighborexchange((x,))
        ...
        >>> ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')
        >>> init()
        >>> net = Net()
        >>> input_x = Tensor(np.ones([3, 3]), dtype = ms.float32)
        >>> output = net(input_x)
        >>> print(output)
        [[2. 2.], [2. 2.]]
    """

    @prim_attr_register
    def __init__(self, send_rank_ids, recv_rank_ids, recv_shapes, send_shapes, recv_type,
                 group=GlobalComm.WORLD_COMM_GROUP):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        self.send_rank_ids = send_rank_ids
        self.recv_rank_ids = recv_rank_ids
        self.recv_shapes = recv_shapes
        self.send_shapes = send_shapes
        self.recv_type = recv_type
        self.add_prim_attr('no_eliminate', True)

    def __call__(self, tensor):
        raise NotImplementedError


class AlltoAll(PrimitiveWithInfer):
    r"""
    AlltoAll is a collective operation.

    AlltoAll sends data from the all processes to the all processes in the specified group. It has two phases:

    - The scatter phase: On each process, the operand is split into split_count number of blocks along the
      split_dimensions, and the blocks are scattered to all processes, e.g., the ith block is send to the ith process.
    - The gather phase: Each process concatenates the received blocks along the concat_dimension.

    Note:
        This operator requires a full-mesh network topology, each device has the same vlan id, and the ip & mask are
        in the same subnet, please check the `details \
        <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#注意事项>`_.

    Args:
        split_count (int): On each process, divide blocks into split_count number.
        split_dim (int): On each process, split blocks along the split_dim.
        concat_dim (int): On each process, gather the received blocks along the concat_dimension.
        group (str): The communication group to work on. Default: "GlobalComm.WORLD_COMM_GROUP".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor. If the shape of input tensor is :math:`(x_1, x_2, ..., x_R)`, then the shape of output tensor is
        :math:`(y_1, y_2, ..., y_R)`, where:

        - :math:`y_{split\_dim} = x_{split\_dim} / split\_count`
        - :math:`y_{concat\_dim} = x_{concat\_dim} * split\_count`
        - :math:`y_{other} = x_{other}`.

    Raises:
        TypeError: If group is not a string.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with 8 devices.

        >>> import os
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.alltoall = ops.AlltoAll(split_count = 8, split_dim = -2, concat_dim = -1)
        ...
        ...     def construct(self, x):
        ...         out = self.alltoall(x)
        ...         return out
        ...
        >>> ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')
        >>> init()
        >>> net = Net()
        >>> rank_id = int(os.getenv("RANK_ID"))
        >>> input_x = Tensor(np.ones([1, 1, 8, 1]) * rank_id, dtype = ms.float32)
        >>> output = net(input_x)
        >>> print(output)
        [[[[0. 1. 2. 3. 4. 5. 6. 7.]]]]
    """

    @prim_attr_register
    def __init__(self, split_count, split_dim, concat_dim, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize AlltoAll"""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        validator.check_is_int(split_count, int)
        validator.check_is_int(split_dim, int)
        validator.check_is_int(concat_dim, int)
        self.split_count = split_count
        self.split_dim = split_dim
        self.concat_dim = concat_dim
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('no_eliminate', True)

    def infer_shape(self, x_shape):
        rank_size = get_group_size(_get_group(self.group))
        if self.split_count != rank_size:
            raise ValueError(f"For '{self.name}', the 'split_count' must be equal to 'rank_size', "
                             f"but got 'split_count': {self.split_count}, 'rank_size': {rank_size}.")
        if x_shape[self.split_dim] % self.split_count != 0:
            raise ValueError(f"For '{self.name}', the 'split_count' must be divisible by 'rank_size', "
                             f"but got 'split_count' {self.split_count}, 'rank_size' {x_shape[self.split_dim]}.")
        x_shape[self.concat_dim] = x_shape[self.concat_dim] * self.split_count
        x_shape[self.split_dim] = int(x_shape[self.split_dim] / self.split_count)
        return x_shape

    def infer_dtype(self, x_dtype):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype

    def __call__(self, tensor):
        raise NotImplementedError


class NeighborExchangeV2(Primitive):
    r"""
    NeighborExchangeV2 is a collective communication operation.

    NeighborExchangeV2 sends data from the local rank to ranks in the `send_rank_ids`,
    as while receive data from `recv_rank_ids`. Please refer to
    `Distributed Set Communication Primitives - NeighborExchangeV2 \
    <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#neighborexchangev2>`_
    to learn about how the data is exchanged between neighborhood devices.

    Note:
        This operator requires a full-mesh network topology, each device has the same vlan id, and the ip & mask are
        in the same subnet, please check the `details \
        <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#注意事项>`_.

    Args:
        send_rank_ids (list(int)): Ranks which the data is sent to. 8 rank_ids represents 8 directions, if one
                                   direction is not send to , set it -1.
        recv_rank_ids (list(int)): Ranks which the data is received from. 8 rank_ids represents 8 directions,
                                   if one direction is not recv from , set it -1.
        send_lens (list(int)): Data lens which send to the send_rank_ids, 4 numbers represent the lens of
                               [send_top, send_bottom, send_left, send_right].
        recv_lens (list(int)): Data lens which received from recv_rank_ids, 4 numbers represent the lens of
                               [recv_top, recv_bottom, recv_left, recv_right].
        data_format (str): Data format, only support NCHW now.
        group (str, optional): The communication group to work on. Default: "GlobalComm.WORLD_COMM_GROUP", which means
                     "hccl_world_group" in Ascend, and "nccl_world_group" in GPU.

    Inputs:
        - **input_x** (Tensor) - The Tensor before being exchanged. It has a shape of :math:`(N, C, H, W)`.

    Outputs:
        The Tensor after being exchanged. If input shape is :math:`(N, C, H, W)`, output shape is
        :math:`(N, C, H+recv\_top+recv\_bottom, W+recv\_left+recv\_right)`.

    Raises:
        TypeError: If `group` is not a string or any one of `send_rank_ids`,
            `recv_rank_ids`, `send_lens`, `recv_lens` is not a list.
        ValueError: If `send_rank_ids` or `recv_rank_ids` has value less than -1 or has repeated values.
        ValueError: If `send_lens`, `recv_lens` has value less than 0.
        ValueError: If `data_format` is not "NCHW".

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with 2 devices.

        >>> import os
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.neighborexchangev2 = ops.NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
        ...                                                          send_lens=[0, 1, 0, 0],
        ...                                                          recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
        ...                                                          recv_lens=[0, 1, 0, 0],
        ...                                                          data_format="NCHW")
        ...
        ...     def construct(self, x):
        ...         out = self.neighborexchangev2(x)
        ...         return out
        ...
        >>> ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')
        >>> init()
        >>> input_x = Tensor(np.ones([1, 1, 2, 2]), dtype = ms.float32)
        >>> net = Net()
        >>> output = net(input_x)
        >>> print(output)
        [[[[1. 1.], [1. 1.], [2. 2.]]]]
    """

    @prim_attr_register
    def __init__(self, send_rank_ids, send_lens, recv_rank_ids, recv_lens, data_format,
                 group=GlobalComm.WORLD_COMM_GROUP):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        self.send_rank_ids = send_rank_ids
        self.recv_rank_ids = recv_rank_ids
        self.send_lens = send_lens
        self.recv_lens = recv_lens
        self.format = data_format
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('no_eliminate', True)

    def __call__(self, tensor):
        raise NotImplementedError


class _MirrorOperator(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do all reduce and mean in backward. It is only for
    internal use of parallel modules and cannot be called by users.

    Args:
        group (str): The communication group to work on. Default: None.
        dev_num (int): The device number of the group. Default: None.
        mean_flag (bool): Whether use mean in backward. Default: None.
    """

    @prim_attr_register
    def __init__(self, group=None, dev_num=None, mean_flag=None):
        """Initialize _MirrorOperator."""
        self.group = group
        self.dev_num = dev_num
        self.mean_flag = mean_flag
        self.add_prim_attr("fusion", 1)
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return x_dtype


mirror = _MirrorOperator()


class _MirrorMiniStepOperator(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do all reduce and mean in backward. It is only for
    internal use of parallel modules and cannot be called by users.

    Args:
        group (str): The communication group to work on. Default: None.
        dev_num (int): The device number of the group. Default: None.
        mean_flag (bool): Whether use mean in backward. Default: None.
        grad_accumulation_step (int): The grad accumulation step. Default: None.
    """

    @prim_attr_register
    def __init__(self, group=None, dev_num=None, mean_flag=None, grad_accumulation_step=None):
        """Initialize _MirrorMiniStepOperator."""
        self.group = group
        self.dev_num = dev_num
        self.mean_flag = mean_flag
        self.grad_accumulation_step = grad_accumulation_step
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape, z_shape):
        return x_shape

    def infer_dtype(self, x_dtype, z_shape):
        return x_dtype


mirror_mini_step = _MirrorMiniStepOperator()


class _VirtualDiv(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do Div in backward.

    Args:
        divisor: float32
    """

    @prim_attr_register
    def __init__(self, divisor=None):
        """Initialize _VirtualDiv."""
        self.divisor = divisor
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return x_dtype


virtual_div = _VirtualDiv()


class _VirtualPipelineEnd(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward and backward, mark end node in pipeline parallel.

    Args:
        divisor: float32
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualPipelineEnd."""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return x_dtype


virtual_pipeline_end = _VirtualPipelineEnd()


class _VirtualAdd(PrimitiveWithInfer):
    """Auto parallel virtual operator. Do nothing in forward, do Add in backward."""

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualAdd."""
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        return x_dtype


class _VirtualDataset(PrimitiveWithInfer):
    """
    Auto parallel virtual dataset operator.

    It would insert VirtualDataset operator in forward computation and be deleted before backward computation.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualDataset."""
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, *args):
        return args

    def infer_dtype(self, *args):
        return args


virtual_dataset = _VirtualDataset()


class _VirtualAssignAdd(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do AssignAdd in backward. It is only for
    internal use of parallel modules and cannot be called by users.

    """

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualAssignAdd."""
        self.add_prim_attr('order_enforce_skip', True)
        self.add_prim_attr('side_effect_backprop_mem', True)

    def infer_shape(self, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        return x_dtype


virtual_assign_add = _VirtualAssignAdd()


class _VirtualAccuGrad(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, return y in backward. It is only for
    internal use of parallel modules and cannot be called by users.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualAccuGrad."""
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        return x_dtype


virtual_accu_grad = _VirtualAccuGrad()


class _MirrorMicroStepOperator(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do all reduce and mean in backward. It is only for
    internal use of parallel modules and cannot be called by users.

    Args:
        group (str): The communication group to work on. Default: None.
        dev_num (int): The device number of the group. Default: None.
        mean_flag (bool): Whether use mean in backward. Default: None.
    """

    @prim_attr_register
    def __init__(self, group=None, dev_num=None, mean_flag=None):
        """Initialize _MirrorMicroStepOperator."""
        self.group = group
        self.dev_num = dev_num
        self.mean_flag = mean_flag
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape, z_shape):
        return x_shape

    def infer_dtype(self, x_dtype, z_shape):
        return x_dtype


class _VirtualOutput(PrimitiveWithInfer):
    """
    Auto parallel virtual out operator.

    It would insert VirtualOutput operator in forward computation and be deleted before backward computation.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualOutput."""
        self.add_prim_attr('order_enforce_skip', True)

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return x_dtype


class _GetTensorSlice(PrimitiveWithInfer):
    """
    Gets tensor slice by device matrix and tensor map.

    Args:
        dev_mat (tuple): The device matrix of the slice tensor.
        tensor_map (tuple): The tensor map of the slice tensor.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _GetTensorSlice."""
        self.add_prim_attr('order_enforce_skip', True)

    def infer_value(self, x, dev_mat, tensor_map):
        from mindspore.parallel._tensor import _load_tensor
        validator.check_value_type("dev_mat", dev_mat, [tuple], self.name)
        validator.check_value_type("tensor_map", tensor_map, [tuple], self.name)
        return Tensor(_load_tensor(x, dev_mat, tensor_map))
