# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from mindspore import _checkparam as validator
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
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            For the CPU device, users need to write a dynamic cluster startup script, please see the `Dynamic Cluster
            Startup <https://www.mindspore.cn/tutorials/experts/en/master/parallel/dynamic_cluster.html>`_ .

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
    default_target_dtypes = (mstype.int8, mstype.int32, mstype.float16, mstype.float32, mstype.bfloat16)
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
    Reduces tensors across all devices in such a way that all devices will get the same final result,
    returns the tensor which is all reduced.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        op (str, optional): Specifies an operation used for element-wise reductions, like sum, prod, max, and min.
                  On the CPU, only 'sum' is supported. Default: ``ReduceOp.SUM`` .
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
                  means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape of the input, i.e., :math:`(x_1, x_2, ..., x_R)`.
        The contents depend on the specified operation.

    Raises:
        TypeError: If any of `op` and `group` is not a str or the input's dtype is bool.
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
        >>> from mindspore import Tensor
        >>> from mindspore.ops import ReduceOp
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
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

    Tutorial Examples:
        - `Distributed Set Communication Primitives - AllReduce
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#allreduce>`_

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


class Reduce(PrimitiveWithInfer):
    """
    Reduces tensors across the processes in the specified communication group, sends the result
    to the target dest_rank(local rank), and returns the tensor which is sent to the target process.

    Note:
        Only process with destination rank receives the reduced output.
        Support PyNative mode and Graph mode, but Graph mode only supports scenes with a graph compilation level of O0.
        Other processes only get a tensor with shape [1], which has no mathematical meaning.

    Args:
        dest_rank (int): The target process(local rank) in the specific group that receives the reduced output.
        op (str, optional): Specifies an operation used for element-wise reductions, like sum, prod, max, and min.
                  On the CPU, only 'sum' is supported. Default: ``ReduceOp.SUM`` .
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
                  means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor. Return the tensor in the specific rank of the process after reduction.
        The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If the type of the first input parameter is not Tensor,
                or any of `op` and `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method without any third-party
            or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> from mindspore import ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> # Launch 4 processes.
        >>> init()
        >>> class ReduceNet(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.reduce = ops.Reduce(dest_rank=1)
        >>>
        >>>     def construct(self, x):
        >>>         out = self.reduce(x)
        >>>         return out
        >>> input = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = ReduceNet()
        >>> output = net(input)
        >>> print(output)
        Process with rank 1: [[4. 4. 4. 4. 4. 4. 4. 4.]
                             [4. 4. 4. 4. 4. 4. 4. 4.]],
        Other proesses: [0.].
    """

    @prim_attr_register
    def __init__(self, dest_rank, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        validator.check_value_type('op', op, (type(ReduceOp.SUM),), self.name)
        self.dest_rank = dest_rank
        self.op = op
        self.group = _get_group(group)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('dest_rank', dest_rank)

    def infer_shape(self, x_shape):
        # The process with dest_rank returns the reduced output.
        # Other processes only gets a tensor with shape [1], which has no mathematical meaning.
        if self.dest_rank == get_rank():
            return x_shape
        return [1]

    def infer_dtype(self, x_dtype):
        return x_dtype


class AllGather(PrimitiveWithInfer):
    """
    Gathers tensors from the specified communication group and returns the tensor which is all gathered.

    Note:
        - The tensors must have the same shape and format in all processes of the collection.

    Args:
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
            means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor. If the number of devices in the group is N,
        then the shape of output is :math:`(N, x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `group` is not a str.
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

    Tutorial Examples:
        - `Distributed Set Communication Primitives - AllGather
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#allgather>`_

    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize AllGather."""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank = get_rank(_get_group(group))
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank, 'rank_size', self.rank_size, validator.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 0)
        self.add_prim_attr('mean_flag', False)
        self.add_prim_attr('no_eliminate', True)

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
        group (str): The communication group to work on. Default: ``None`` .
        grad_accumulation_step (int): The grad accumulation step. Default: ``None`` .
    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP, grad_accumulation_step=None, mean_flag=None):
        """Initialize _MiniStepAllGather."""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank = get_rank(_get_group(group))
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank, 'rank_size', self.rank_size, validator.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 1)
        self.grad_accumulation_step = grad_accumulation_step
        self.mean_flag = mean_flag
        self.add_prim_attr('order_enforce_skip', True)
        self.add_prim_attr('side_effect_backprop_mem', True)

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
        group (str): The communication group to work on. Default: ``None`` .
    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP, mean_flag=None):
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank_size = 1
        if group != "":
            self.rank = get_rank(_get_group(group))
            self.rank_size = get_group_size(_get_group(group))
            validator.check('rank', self.rank, 'rank_size', self.rank_size, validator.LT, self.name)
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
        group (Union[tuple[int],list[int]]): The rand_ids of communication group to work on. Default: ``None`` .

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
        validator.check_int(len(group), 2, validator.GE, "group size", self.name)
        for r in group:
            validator.check_int_range(r, 0, 7, validator.INC_BOTH, "rank_id", self.name)
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
    Reduces and scatters tensors from the specified communication group
    and returns the tensor which is reduced and scattered.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        op (str, optional): Specifies an operation used for element-wise reductions,
                  like SUM and MAX. Default: ``ReduceOp.SUM`` .
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

    Inputs:
        - **input_x** (Tensor) - Input Tensor, suppose it has a shape :math:`(N, *)`, where `*`
          means any number of additional dimensions. N must be divisible by rank_size.
          rank_size refers to the number of cards in the communication group.

    Outputs:
        Tensor, it has the same dtype as `input_x` with a shape of :math:`(N/rank\_size, *)`.

    Raises:
        TypeError: If any of operation and group is not a string.
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
        >>> from mindspore.ops import ReduceOp
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
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

    Tutorial Examples:
        - `Distributed Set Communication Primitives - ReduceScatter
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#reducescatter>`_

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
                  like sum, max, avg. Default: ``ReduceOp.SUM`` .
        group (Union[tuple[int],list[int]]): The rand_ids of communication group to work on. Default: ``None`` .

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
        validator.check_int(len(group), 2, validator.GE, "group size", self.name)
        for r in group:
            validator.check_int_range(r, 0, 7, validator.INC_BOTH, "rank_id", self.name)
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
        root_rank (int): Specifies the rank(global rank) of the process that broadcast the tensor.
            And only process `root_rank` will broadcast the tensor.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[Tensor], Tensor has the same shape of the input, i.e., :math:`(x_1, x_2, ..., x_R)`.
        The contents depend on the data of the `root_rank` device.

    Raises:
        TypeError: If root_rank is not an integer or group is not a string.

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

            This example should be run with multiple devices.

        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
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

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Broadcast
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#broadcast>`_

    """

    @prim_attr_register
    def __init__(self, root_rank, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize Broadcast."""
        validator.check_value_type('root_rank', root_rank, (int,), self.name)
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        check_hcom_group_valid(group, prim_name=self.name)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('no_eliminate', True)


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
        validator.check_subclass("tensor_in", tensor_in['dtype'], mstype.tensor_type, self.name)
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
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.primitive.html#communication-operator>`_.

        This operator requires a full-mesh network topology, each device has the same vlan id, and the ip & mask are
        in the same subnet, please check the `details \
        <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#notes>`_.

    Args:
        send_rank_ids (list(int)): Ranks which the data is sent to.
        recv_rank_ids (list(int)): Ranks which the data is received from.
        recv_shapes (tuple(list(int))): Data shape which received from recv_rank_ids.
        send_shapes (tuple(list(int))): Data shape which send to the send_rank_ids.
        recv_type (type): Data type which received from recv_rank_ids
        group (str): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

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
        >>> from mindspore import ops
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
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> net = Net()
        >>> input_x = Tensor(np.ones([3, 3]), dtype = ms.float32)
        >>> output = net(input_x)
        >>> print(output)
        [[2. 2.], [2. 2.]]

    Tutorial Examples:
        - `Distributed Set Communication Primitives - NeighborExchange
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#neighborexchange>`_

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
        self.add_prim_attr('group', _get_group(group))
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
        <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#notes>`_.

    Args:
        split_count (int): On each process, divide blocks into split_count number.
        split_dim (int): On each process, split blocks along the split_dim.
        concat_dim (int): On each process, gather the received blocks along the concat_dimension.
        group (str): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

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
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            This example should be run with 8 devices.

        >>> import os
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
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
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> net = Net()
        >>> rank_id = int(os.getenv("RANK_ID"))
        >>> input_x = Tensor(np.ones([1, 1, 8, 1]) * rank_id, dtype = ms.float32)
        >>> output = net(input_x)
        >>> print(output)
        [[[[0. 1. 2. 3. 4. 5. 6. 7.]]]]

    Tutorial Examples:
        - `Distributed Set Communication Primitives - AlltoAll
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#alltoall>`_

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
        if x_shape[self.split_dim] >= 0 and x_shape[self.split_dim] % self.split_count != 0:
            raise ValueError(f"For '{self.name}', the 'x_shape[self.split_dim]' must be divisible by 'split_count', "
                             f"but got 'x_shape[self.split_dim]' {x_shape[self.split_dim]}, "
                             f"'split_count' {self.split_count}.")
        if x_shape[self.concat_dim] >= 0:
            x_shape[self.concat_dim] = x_shape[self.concat_dim] * self.split_count
        if x_shape[self.split_dim] >= 0:
            x_shape[self.split_dim] = int(x_shape[self.split_dim] / self.split_count)
        return x_shape

    def infer_dtype(self, x_dtype):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype


class NeighborExchangeV2(Primitive):
    r"""
    NeighborExchangeV2 is a collective communication operation.

    NeighborExchangeV2 sends data from the local rank to ranks in the `send_rank_ids`,
    as while receive data from `recv_rank_ids`. Please refer to the tutorial examples
    below to learn about how the data is exchanged between neighborhood devices.

    Note:
        This operator requires a full-mesh network topology, each device has the same vlan id, and the ip & mask are
        in the same subnet, please check the `details \
        <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#notes>`_.

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
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
                     means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

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
            Please see the `rank table Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

            For the CPU device, users need to write a dynamic cluster startup script, please see the `Dynamic Cluster
            Startup <https://www.mindspore.cn/tutorials/experts/en/master/parallel/dynamic_cluster.html>`_ .

            This example should be run with 2 devices.

        >>> import os
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
        >>> import numpy as np
        >>>
        >>> class Net0(nn.Cell):
        ...     def __init__(self):
        ...         super(Net0, self).__init__()
        ...         self.neighbor_exchangev2 = ops.NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
        ...                                                           send_lens=[0, 1, 0, 0],
        ...                                                           recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
        ...                                                           recv_lens=[0, 1, 0, 0], data_format="NCHW")
        ...
        ...     def construct(self, x):
        ...         out = self.neighbor_exchangev2(x)
        ...         return out
        ... class Net1(nn.Cell):
        ...     def __init__(self):
        ...         super(Net1, self).__init__()
        ...         self.neighbor_exchangev2 = ops.NeighborExchangeV2(send_rank_ids=[0, -1, -1, -1, -1, -1, -1, -1],
        ...                                                           send_lens=[1, 0, 0, 0],
        ...                                                           recv_rank_ids=[0, -1, -1, -1, -1, -1, -1, -1],
        ...                                                           recv_lens=[1, 0, 0, 0], data_format="NCHW")
        ...
        ...     def construct(self, x):
        ...         out = self.neighbor_exchangev2(x)
        ...         return out
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> rank_id = int(os.getenv("RANK_ID"))
        >>> if (rank_id % 2 == 0):
        >>>     input_x = ms.Tensor(np.ones([1, 1, 2, 2]), dtype = ms.float32)
        >>>     net = Net0()
        >>>     output = net(input_x)
        >>>     print(output)
        >>> else:
        >>>     input_x = ms.Tensor(np.ones([1, 1, 2, 2]) * 2, dtype = ms.float32)
        >>>     net = Net1()
        >>>     output = net(input_x)
        >>>     print(output)
        [[[[1. 1.], [1. 1.], [2. 2.]]]]

    Tutorial Examples:
        - `Distributed Set Communication Primitives - NeighborExchangeV2
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#neighborexchangev2>`_

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
        self.rank_size = get_group_size(_get_group(group))
        for rank_id in send_rank_ids:
            if rank_id != -1:
                validator.check_number_range(rank_id, 0, self.rank_size, validator.INC_LEFT, int,
                                             "rank_id in send_rank_ids")
        for rank_id in recv_rank_ids:
            if rank_id != -1:
                validator.check_number_range(rank_id, 0, self.rank_size, validator.INC_LEFT, int,
                                             "rank_id in recv_rank_ids")

    def __call__(self, tensor):
        raise NotImplementedError


class CollectiveScatter(Primitive):
    r"""
    Scatter tensor evently across the processes in the specified communication group.

    Note:
        The interface behavior only support Tensor input and scatter evenly.
        Only the tensor in process `src_rank` (global rank) will do scatter.

    Args:
        src_rank (int, optional): Specifies the rank of the process that send the tensor.
            And only process `src_rank` will send the tensor.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Inputs:
        - **input_x** (Tensor) - The input tensor to be scattered. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, the shape of output is :math:`(x_1/src\_rank, x_2, ..., x_R)`. The dimension 0 of data is equal to
        the dimension of input tensor divided by `src`, and the other dimension keep the same.

    Raises:
        TypeError: If `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.
        ValueError: If the local rank id of the calling process in the group
            is larger than the group's rank size.

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
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init, get_rank
        >>> from mindspore import ops
        >>> # Launch 2 processes.
        >>> init()
        >>> class CollectiveScatterNet(nn.Cell):
        >>>     def __init__(self):
        >>>         super(CollectiveScatter, self).__init__()
        >>>         self.collective_scatter = ops.CollectiveScatter(src_rank=0)
        >>>
        >>>     def construct(self, x):
        >>>         return self.collective_scatter(x)
        >>>
        >>> input = Tensor(np.arange(8).reshape([4, 2]).astype(np.float32))
        >>> net = CollectiveScatterNet()
        >>> output = net(input)
        >>> print(output)
        Process with rank 0: [[0. 1.],
                              [2. 3.]]
        Process with rank 1: [[4. 5.],
                              [6. 7.]]

    Tutorial Examples:
        - `Distributed Set Communication Primitives - CollectiveScatter
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#reducescatter>`_

    """

    @prim_attr_register
    def __init__(self, src_rank=0, group=GlobalComm.WORLD_COMM_GROUP):
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank_id = get_rank(_get_group(group))
        self.src_rank = src_rank
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank_id, 'rank_size', self.rank_size, validator.LT, self.name)
        self.add_prim_attr('rank_id', self.rank_id)
        self.add_prim_attr('src_rank', self.src_rank)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))


class CollectiveGather(Primitive):
    r"""
    Gathers tensors from the specified communication group. The operation will gather the tensor
    from processes according to dimension 0.

    Note:
        Only the tensor in process `dest_rank` (global rank) will keep the gathered tensor. The other process
        will keep a tensor with shape [1], which has no mathematical meaning.

    Args:
        dest_rank(int): Specifies the rank of the process that receive the tensor.
            And only process `dest_rank` will receive the gathered tensor.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Inputs:
        - **input_x** (Tensor) - The tensor to be gathered. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, the shape of output is :math:`(\sum x_1, x_2, ..., x_R)`. The dimension 0 of data is equal to
        sum of the dimension of input tensor, and the other dimension keep the same.

    Raises:
        TypeError: If `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.
        ValueError: If the local rank id of the calling process in the group
                    is larger than the group's rank size.

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

            This example should be run with 4 devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> # Launch 2 processes.
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> class CollectiveGatherNet(nn.Cell):
        ...     def __init__(self):
        ...         super(CollectiveGatherNet, self).__init__()
        ...         self.collective_gather = ops.CollectiveGather(dest_rank=0)
        ...
        ...     def construct(self, x):
        ...         return self.collective_gather(x)
        ...
        >>> input = Tensor(np.arange(4).reshape([2, 2]).astype(np.float32))
        >>> net = CollectiveGatherNet()
        >>> output = net(input)
        >>> print(output)
        Process with rank 0: [[0. 1.],
                              [2. 3.],
                              [0. 1.],
                              [2. 3.]]
        Process with rank 1: [0.]

    Tutorial Examples:
        - `Distributed Set Communication Primitives - CollectiveGather
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#collectivegather>`_

    """

    @prim_attr_register
    def __init__(self, dest_rank, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize Gather."""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank_id = get_rank(_get_group(group))
        self.dest_rank = dest_rank
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank_id, 'rank_size', self.rank_size, validator.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('dest_rank', self.dest_rank)
        self.add_prim_attr('rank_id', self.rank_id)


class Barrier(PrimitiveWithInfer):
    """
    Synchronizes all processes in the specified group. Once the process call this operation, it will be blocked until
    all processes call this operation. After all processes finish calling the operations, the blocked processes
    will be waken and continue their task.

    Args:
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Raises:
        TypeError: If `group` is not a str.
        RuntimeError: If backend is invalid, or distributed initialization fails.
        ValueError: If the local rank id of the calling process in the group
                    is larger than the group's rank size.

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
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> # Launch 4 processes.
        >>> init()
        >>> class BarrierNet(nn.Cell):
        >>>     def __init__(self):
        >>>         super(BarrierNet, self).__init__()
        >>>         self.barrier = ops.Barrier()
        >>>
        >>>     def construct(self):
        >>>         self.barrier()
        >>> net = BarrierNet()
        >>> net()

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Barrier
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#barrier>`_

    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        self.group = group
        self.add_prim_attr("side_effect_mem", True)

    def infer_shape(self):
        return [1]

    def infer_dtype(self):
        return mstype.float32


class Send(PrimitiveWithInfer):
    """
    Send tensors to the specified dest_rank.

    Note:
        Send and Receive must be used in combination and have same sr_tag.

    Args:
        sr_tag (int): The tag to identify the send/recv message. The message will
                      be received by the Receive op with the same "sr_tag".
        dest_rank (int): A required integer identifying the destination rank.
        group_back (str, optional): The communication group for backpropagation.
                                    Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.
        ValueError: If the local rank id of the calling process in the group
                    is larger than the group's rank size.

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

        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>>
        >>> init()
        >>> class SendNet(nn.Cell):
        >>>     def __init__(self):
        >>>         super(SendNet, self).__init__()
        >>>         self.depend = ops.Depend()
        >>>         self.send = ops.Send(st_tag=0, dest_rank=8, group="hccl_world_group")
        >>>
        >>>     def construct(self, x):
        >>>         out = self.depend(x, self.send(x))
        >>>         return out
        >>>
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_)

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Send
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#send>`_

    """

    @prim_attr_register
    def __init__(self, sr_tag, dest_rank, group=GlobalComm.WORLD_COMM_GROUP, group_back=GlobalComm.WORLD_COMM_GROUP):
        self.rank = dest_rank
        self.sr_tag = sr_tag
        self.group = group
        self.add_prim_attr("no_eliminate", True)

    def infer_shape(self, x_shape):
        self.add_prim_attr("shape", x_shape)
        return x_shape

    def infer_dtype(self, x_dtype):
        return x_dtype


class Receive(PrimitiveWithInfer):
    """
    Receive tensors from src_rank.

    Note:
        Send and Receive must be used in combination and have same sr_tag.

    Args:
        sr_tag (int): A required integer identifying the send/recv message tag. The message will
                      will be send by the Send op with the same "sr_tag".
        src_rank (int): A required integer identifying the source rank.
        shape (list[int]): A required list identifying the shape of the tensor to be received.
        dtype (Type): A required Type identifying the type of the tensor to be received. The supported types:
                       int8/int16/int32/float16/float32.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.
        group_back (str, optional): The communication group for backpropagation.
                                    Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Outputs:
        Tensor, output has the same shape as the Tensor sent by `Send` operation.

    Raises:
        TypeError: If `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.
        ValueError: If the local rank id of the calling process in the group
                    is larger than the group's rank size.

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

        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>>
        >>> init()
        >>> class ReceiveNet(nn.Cell):
        >>>     def __init__(self):
        >>>         super(ReceiveNet, self).__init__()
        >>>         self.recv = ops.Receive(sr_tag=0, src_rank=0, shape=[2, 8], dtype=ms.float32,
        >>>                               group="hccl_world_group")
        >>>
        >>>     def construct(self):
        >>>         out = self.recv()
        >>>         return out
        >>>
        >>> net = Net()
        >>> output = net()

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Receive
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#receive>`_

    """

    @prim_attr_register
    def __init__(self, sr_tag, src_rank, shape, dtype, group=GlobalComm.WORLD_COMM_GROUP,
                 group_back=GlobalComm.WORLD_COMM_GROUP):
        self.rank = src_rank
        self.tag = sr_tag
        self.shape = shape
        self.dtype = dtype
        self.group = group
        self.add_prim_attr("no_eliminate", True)
        valid_type = [mstype.float16, mstype.float32, mstype.float64, mstype.bfloat16,
                      mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                      mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64]
        args = {"dtype": dtype}
        validator.check_scalar_or_tensor_types_same(args, valid_type, self.name)

    def infer_shape(self, x_shape=None):
        return self.get_attr_dict()['shape']

    def infer_dtype(self, x_dtype=None):
        return self.get_attr_dict()['dtype']


class _MirrorOperator(PrimitiveWithInfer):
    """
    Auto parallel virtual operator. Do nothing in forward, do all reduce and mean in backward. It is only for
    internal use of parallel modules and cannot be called by users.

    Args:
        group (str): The communication group to work on. Default: ``None`` .
        dev_num (int): The device number of the group. Default: ``None`` .
        mean_flag (bool): Whether use mean in backward. Default: ``None`` .
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
        group (str): The communication group to work on. Default: ``None`` .
        dev_num (int): The device number of the group. Default: ``None`` .
        mean_flag (bool): Whether use mean in backward. Default: ``None`` .
        grad_accumulation_step (int): The grad accumulation step. Default: ``None`` .
    """

    @prim_attr_register
    def __init__(self, group=None, dev_num=None, mean_flag=None, grad_accumulation_step=None):
        """Initialize _MirrorMiniStepOperator."""
        self.group = group
        self.dev_num = dev_num
        self.mean_flag = mean_flag
        self.grad_accumulation_step = grad_accumulation_step
        self.add_prim_attr('order_enforce_skip', True)
        self.add_prim_attr('side_effect_backprop_mem', True)

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


class _VirtualConverterEnd(PrimitiveWithInfer):
    """
    Auto parallel virtual operator.
    """

    @prim_attr_register
    def __init__(self, input_nums):
        """Initialize _VirtualConverterEnd."""
        self.input_nums = input_nums

    def infer_shape(self, *args):
        return (args[0][0] * self.input_nums,) + tuple(args[0][1:])

    def infer_dtype(self, *args):
        return args[0]

class _VirtualConverterBegin(PrimitiveWithInfer):
    """
    Auto parallel virtual operator.
    """

    @prim_attr_register
    def __init__(self, output_nums):
        """Initialize _VirtualConverterBegin."""
        self.output_nums = output_nums

    def infer_shape(self, arg):
        new_arg = (arg[0] / self.output_nums,) + tuple(arg[1:])
        return (new_arg,) * self.output_nums

    def infer_dtype(self, arg):
        return (arg,) * self.output_nums


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
        group (str): The communication group to work on. Default: ``None`` .
        dev_num (int): The device number of the group. Default: ``None`` .
        mean_flag (bool): Whether use mean in backward. Default: ``None`` .
    """

    @prim_attr_register
    def __init__(self, group=None, dev_num=None, mean_flag=None):
        """Initialize _MirrorMicroStepOperator."""
        self.group = group
        self.dev_num = dev_num
        self.mean_flag = mean_flag
        self.add_prim_attr('order_enforce_skip', True)
        self.add_prim_attr('side_effect_backprop_mem', True)

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

    def infer_value(self, x, dev_mat, tensor_map, slice_shape, full_shape):
        from mindspore.parallel._tensor import _load_tensor
        validator.check_value_type("dev_mat", dev_mat, [tuple], self.name)
        validator.check_value_type("tensor_map", tensor_map, [tuple], self.name)
        tensor_slice = _load_tensor(x, dev_mat, tensor_map, full_shape)
        if tensor_slice.shape != slice_shape:
            tensor_slice = tensor_slice.reshape(slice_shape)
        return Tensor(tensor_slice, x.dtype)


class BatchISendIRecv(PrimitiveWithInfer):
    """
    Batch send and recv tensors asynchronously.

    Note:
        - The ``isend`` and ``irecv`` in ``op_types`` between ranks need to match each other.
        - ``isend`` and ``irecv`` in a batch can only be used in the same communication group.

    Args:
        op_types(Union[tuple[str], list[str]]): "isend" or "irecv" to indicate the order and number of communication.
        remote_ranks(Union[tuple[int], list[int]]): src or dst rank that matches the op_types.
        receive_shapes(Union[tuple[int], list[int]]): receive tensor shapes that matches "irecv" in op_types.
        receive_types(Union[tuple[mindspore.dtype], list[mindspore.dtype]]): receive tensor dtype
          that matches "irecv" in op_types.
        group (str): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``, which
          means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Inputs:
        - **input_x** (Union[tuple[Tensor], list[Tensor], tuple(None)]) -
          The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple(Tensor). Output tensors is corresponding to ``op_types``:
        At ``"isend"`` position, output tensor is a fake tensor with scalar, which has no meaning.
        At ``"irecv"`` position, output tensor is a tensor received from remote end.


    Raises:
        TypeError: If ``group`` is not a str.
        TypeError: If ``op_types``, ``receive_shapes``, ``receive_dtypes``, ``remote_ranks`` are not tuple or list.
        ValueError: If the length of ``receive_shapes`` and ``receive_dtypes`` are not the same.
        ValueError: If the length of ``op_types`` and ``remote_ranks`` are not the same.
        RuntimeError: If the length of input tensors and ``"isend"`` count are not the same.

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
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init, get_rank
        >>> from mindspore import Tensor
        >>>
        >>> init()
        >>> rank = get_rank()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         if rank == 0:
        ...             remote_rank = [1, 1]
        ...         else:
        ...             remote_rank = [0, 0]
        ...         self.batchisendirecv = ops.BatchISendIRecv(("isend", "irecv"), remote_rank, [()], (ms.float32,))
        ...
        ...     def construct(self, x):
        ...         if isinstance(x, Tensor):
        ...             x = (x,)
        ...         return self.batchisendirecv(x)
        ...
        >>> send_x = Tensor(rank + 1).astype(ms.float32)
        >>> net = Net()
        >>> output = net(send_x)
        >>> print(output)
        rank 0:
        (Tensor(shape=[], dtype=Float32, value= 0), Tensor(shape=[], dtype=Float32, value= 2))
        rank 1:
        (Tensor(shape=[], dtype=Float32, value= 0), Tensor(shape=[], dtype=Float32, value= 1))

    Tutorial Examples:
        - `Distributed Set Communication Primitives - BatchISendIRecv
          <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html#allgather>`_

    """

    @prim_attr_register
    def __init__(self, op_types, remote_ranks, receive_shapes=None,
                 receive_dtypes=None, group=GlobalComm.WORLD_COMM_GROUP):
        if receive_shapes is None:
            receive_shapes = ()
        else:
            validator.check_value_type("receive_shapes", receive_shapes, [tuple, list], self.name)

        if receive_dtypes is None:
            receive_dtypes = ()
        else:
            validator.check_value_type("receive_dtypes", receive_dtypes, [tuple, list], self.name)

        validator.check_value_type("op_types", op_types, [tuple, list], self.name)
        validator.check_value_type("remote_ranks", remote_ranks, [tuple, list], self.name)

        if len(receive_shapes) != len(receive_dtypes):
            raise ValueError("length of receive_shapes and receive_shapes must be the same, "
                             f"but got receive_shapes: {len(receive_shapes)} "
                             f" and receive_shapes: {receive_dtypes}")

        if len(op_types) != len(remote_ranks):
            raise ValueError("length of op_types and remote_ranks must be the same.")

        if group is None:
            group = GlobalComm.WORLD_COMM_GROUP
        self.add_prim_attr('group', group)
        self.add_prim_attr('op_types', op_types)
        self.add_prim_attr('remote_ranks', remote_ranks)
        self.add_prim_attr('receive_shapes', receive_shapes)
        self.add_prim_attr('receive_dtypes', receive_dtypes)
        self.add_prim_attr('no_eliminate', True)


class AlltoAllV(PrimitiveWithInfer):
    """
    AllToAll which support uneven split.

    Note:
        - Only support flatten tensor as input. input tensor should be flattened and
          concatenated before call this primitive.

    Args:
        send_numel_list(Union[tuple[int], list[int]]): split numel to scatter to different remote rank.
        recv_numel_list(Union[tuple[int], list[int]]): split numel to gather from different remote rank.
        group (str): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``, which
          means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Inputs:
        - **input_x** (Tensor) - flatten tensor to scatter. The shape of tensor is :math:`(x_1)`.

    Outputs:
        Tensor. flattened and concatenated tensor gather from remote ranks.
        If gather result is empty, it will return a Tensor with value 0, which has no actual meaning.

    Raises:
        TypeError: If 'send_numel_list'  or 'recv_numel_list' is not type of tuple and list.

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
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init, get_rank
        >>> from mindspore import Tensor
        >>>
        >>> init()
        >>> rank = get_rank()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         if rank == 0:
        ...             self.all_to_all = ops.AlltoAllV([1, 2], [1, 2])
        ...         else:
        ...             self.all_to_all = ops.AlltoAllV([2, 1], [2, 1])
        ...
        ...     def construct(self, x):
        ...         return self.all_to_all(x)
        ...
        >>> if rank == 0:
        >>>    send_tensor = Tensor([0, 1, 2.])
        >>> elif rank == 1:
        >>>    send_tensor = Tensor([3, 4, 5.])
        >>> net = Net()
        >>> output = net(send_tensor)
        >>> print(output)
        rank 0:
        [0. 3. 4]
        rank 1:
        [1. 2. 5]

    """

    @prim_attr_register
    def __init__(self, send_numel_list, recv_numel_list, group=None):
        validator.check_value_type("send_numel_list", send_numel_list, [tuple, list], self.name)
        validator.check_value_type("recv_numel_list", recv_numel_list, [tuple, list], self.name)
        if group is None:
            group = GlobalComm.WORLD_COMM_GROUP
        self.add_prim_attr('group', group)
        self.add_prim_attr('send_numel_list', send_numel_list)
        self.add_prim_attr('recv_numel_list', recv_numel_list)
