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

"""comm_ops"""

from mindspore.common import Tensor
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...communication.management import get_rank, get_group_size, GlobalComm, _get_group
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register
from ...common.api import context

class ReduceOp:
    """
    Operation options for reducing tensors.

    There are four kinds of operation options, "SUM", "MAX", "MIN", and "PROD".

    - SUM: Take the sum.
    - MAX: Take the maximum.
    - MIN: Take the minimum.
    - PROD: Take the product.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    PROD = "prod"


target_dtypes = (mstype.int8, mstype.int32, mstype.float16, mstype.float32)

def check_hcom_group_valid(group):
    if context.get_context("mode") == context.PYNATIVE_MODE and \
            context.get_context("device_target") == "Ascend" and \
            group != GlobalComm.WORLD_COMM_GROUP:
        raise RuntimeError("Only hccl_world_group is supported in Pynative mode, but got {}".format(group))

class AllReduce(PrimitiveWithInfer):
    """
    Reduces the tensor data across all devices in such a way that all devices will get the same final result.

    Note:
        The operation of AllReduce does not support "prod" currently.
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        op (str): Specifies an operation used for element-wise reductions,
                  like sum, max, and min. Default: ReduceOp.SUM.
        group (str): The communication group to work on. Default: "hccl_world_group".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape of the input, i.e., :math:`(x_1, x_2, ..., x_R)`.
        The contents depend on the specified operation.

    Raises:
        TypeError: If any of `op` and `group` is not a str,
                   or fusion is not an integer, or the input's dtype is bool.
        ValueError: If the `op` is "prod".

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations.comm_ops import ReduceOp
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.operations as ops
        >>>
        >>> init()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.allreduce_sum = ops.AllReduce(ReduceOp.SUM, group="nccl_world_group")
        ...
        ...     def construct(self, x):
        ...         return self.allreduce_sum(x)
        ...
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_)
        >>> print(output)
        [[4. 5. 6. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
        if not isinstance(op, type(ReduceOp.SUM)):
            raise TypeError("The operation of AllReduce should be str.")
        if not isinstance(_get_group(group), str):
            raise TypeError("The group of AllReduce should be str.")
        check_hcom_group_valid(group)
        self.op = op
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 0)
        self.add_prim_attr('index', 0)

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, target_dtypes, self.name)
        return x_dtype


class AllGather(PrimitiveWithInfer):
    """
    Gathers tensors from the specified communication group.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        group (str): The communication group to work on. Default: "hccl_world_group".

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
        >>> # This example should be run with two devices. Refer to the tutorial > Distributed Training on mindspore.cn.
        >>> import numpy as np
        >>> import mindspore.ops.operations as ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor, context
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> init()
        ... class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.allgather = ops.AllGather()
        ...
        ...     def construct(self, x):
        ...         return self.allgather(x)
        ...
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_)
        >>> print(output)
        [[1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1. 1. 1.]]
    """

    @prim_attr_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank = get_rank(_get_group(group))
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank, 'rank_size', self.rank_size, Rel.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 0)

    def infer_shape(self, x_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.rank_size
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, target_dtypes, self.name)
        return x_dtype

    def __call__(self, tensor):
        raise NotImplementedError


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
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.rank = get_rank(_get_group(group))
        self.rank_size = get_group_size(_get_group(group))
        validator.check('rank', self.rank, 'rank_size', self.rank_size, Rel.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 1)
        self.grad_accumulation_step = grad_accumulation_step
        self.mean_flag = mean_flag

    def infer_shape(self, x_shape, z_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.rank_size
        return x_shape

    def infer_dtype(self, x_dtype, z_shape):
        validator.check_tensor_dtype_valid('x', x_dtype, target_dtypes, self.name)
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
        group (Union[tuple[int],list[int]]): The rand_ids of communication group to work on.

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
        if group is None:
            raise ValueError(f"For '{self.name}' group must be set.")
        validator.check_value_type('group', group, (tuple, list), self.name)
        validator.check_int(len(group), 2, Rel.GE, "group size", self.name)
        for r in group:
            validator.check_int_range(r, 0, 7, Rel.INC_BOTH, "rank_id", self.name)
            validator.check_value_type("rank_id", r, (int,), self.name)
        self.group_size = len(group)
        self.add_prim_attr('group', group)

    def infer_shape(self, x_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.group_size
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, target_dtypes, self.name)
        return x_dtype

    def __call__(self, tensor):
        raise NotImplementedError


class ReduceScatter(PrimitiveWithInfer):
    """
    Reduces and scatters tensors from the specified communication group.

    Note:
        The back propagation of the op is not supported yet. Stay tuned for more.
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        op (str): Specifies an operation used for element-wise reductions,
                  like SUM, MAX, AVG. Default: ReduceOp.SUM.
        group (str): The communication group to work on. Default: "hccl_world_group".

    Raises:
        TypeError: If any of operation and group is not a string.
        ValueError: If the first dimension of the input cannot be divided by the rank size.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> # This example should be run with two devices. Refer to the tutorial > Distributed Training on mindspore.cn.
        >>> from mindspore import Tensor, context
        >>> from mindspore.communication import init
        >>> from mindspore.ops.operations.comm_ops import ReduceOp
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.operations as ops
        >>> import numpy as np
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE)
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
        validator.check_value_type('op', op, (type(ReduceOp.SUM),), self.name)
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.op = op
        self.rank_size = get_group_size(_get_group(group))
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', _get_group(group))
        self.add_prim_attr('fusion', 0)

    def infer_shape(self, x_shape):
        if x_shape[0] % self.rank_size != 0:
            raise ValueError(f"For '{self.name}' the first dimension of x should be divided by rank_size.")
        x_shape[0] = int(x_shape[0]/self.rank_size)
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, target_dtypes, self.name)
        return x_dtype

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
        group (Union[tuple[int],list[int]]): The rand_ids of communication group to work on.

    Raises:
        TypeError: If op is not a string and group is not a list nor tuple,
                   or elements of group are not int.
        ValueError: If the first dimension of input can not be divided by group size,
                    or group is not set, or rank_id not in [0, 7].
    """
    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM, group=None):
        if group is None:
            raise ValueError(f"For '{self.name}' group must be set.")
        validator.check_value_type('op', op, (type(ReduceOp.SUM),), self.name)
        validator.check_value_type('group', group, (tuple, list), self.name)
        validator.check_int(len(group), 2, Rel.GE, "group size", self.name)
        for r in group:
            validator.check_int_range(r, 0, 7, Rel.INC_BOTH, "rank_id", self.name)
            validator.check_value_type("rank_id", r, (int,), self.name)
        self.op = op
        self.group_size = len(group)
        self.add_prim_attr('group', group)

    def infer_shape(self, x_shape):
        if x_shape[0] % self.group_size != 0:
            raise ValueError(f"For '{self.name}' the first dimension of x should be divided by group_size.")
        x_shape[0] = int(x_shape[0]/self.group_size)
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, target_dtypes, self.name)
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
        group (str): The communication group to work on. Default: "hccl_world_group".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape of the input, i.e., :math:`(x_1, x_2, ..., x_R)`.
        The contents depend on the data of the `root_rank` device.

    Raises:
        TypeError: If root_rank is not a integer or group is not a string.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> # This example should be run with multiple processes.
        >>> # Please refer to the tutorial > Distributed Training on mindspore.cn.
        >>> from mindspore import Tensor
        >>> from mindspore import context
        >>> from mindspore.communication import init
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.operations as ops
        >>> import numpy as np
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> init()
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.broadcast = ops.Broadcast(1)
        ...
        ...     def construct(self, x):
        ...         return self.broadcast((x,))
        ...
        >>> input_ = Tensor(np.ones([2, 4]).astype(np.int32))
        >>> net = Net()
        >>> output = net(input_)
        >>> print(output)
        (Tensor(shape[2,4], dtype=Int32, value=
        [[1, 1, 1, 1],
         [1, 1, 1, 1]]),)
    """

    @prim_attr_register
    def __init__(self, root_rank, group=GlobalComm.WORLD_COMM_GROUP):
        validator.check_value_type('root_rank', root_rank, (int,), self.name)
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        check_hcom_group_valid(group)
        self.add_prim_attr('group', _get_group(group))

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        if not isinstance(x_dtype, tuple):
            raise TypeError(f"{self.name}'s input should be a tuple!")
        for _ele in x_dtype:
            validator.check_tensor_dtype_valid('x', _ele, target_dtypes, self.name)
        return x_dtype


class AllSwap(PrimitiveWithCheck):
    """
    AllSwap is a collective operation.

    AllSwap sends data from the all processes to the all processes in the specified group. It has two phases:

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
        """Initialize AllSwap"""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.init_prim_io_names(inputs=['tensor_in', 'send_size', 'recv_size'], outputs=['tensor_out'])
        self.add_prim_attr('group', _get_group(group))

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


class _AlltoAll(PrimitiveWithInfer):
    """
    AlltoAll is a collective operation.

    AlltoAll sends data from the all processes to the all processes in the specified group. It has two phases:

    - The scatter phase: On each process, the operand is split into split_count number of blocks along the
      split_dimensions, and the blocks are scattered to all processes, e.g., the ith block is send to the ith process.
    - The gather phase: Each process concatenates the received blocks along the concat_dimension.

    Note:
        The tensors must have the same shape and format in all processes of the collection.

    Args:
        split_count (int): On each process, divide blocks into split_count number.
        split_dim (int): On each process, split blocks along the split_dim.
        concat_dim (int): On each process, gather the received blocks along the concat_dimension.
        group (str): The communication group to work on. Default: "hccl_world_group".

    Raises:
        TypeError: If group is not a string.
    """

    @prim_attr_register
    def __init__(self, split_count, split_dim, concat_dim, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize AlltoAll"""
        validator.check_value_type('group', _get_group(group), (str,), self.name)
        self.split_count = split_count
        self.split_dim = split_dim
        self.concat_dim = concat_dim
        self.add_prim_attr('group', _get_group(group))

    def infer_shape(self, x_shape):
        x_shape[self.concat_dim] = x_shape[self.concat_dim] * self.split_count
        x_shape[self.split_dim] = int(x_shape[self.split_dim] / self.split_count)
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, target_dtypes, self.name)
        return x_dtype

    def __call__(self, tensor):
        return


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
        self.group = group
        self.dev_num = dev_num
        self.mean_flag = mean_flag
        self.add_prim_attr("fusion", 1)

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
        self.group = group
        self.dev_num = dev_num
        self.mean_flag = mean_flag
        self.grad_accumulation_step = grad_accumulation_step

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
        self.divisor = divisor

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return x_dtype


virtual_div = _VirtualDiv()


class _VirtualAdd(PrimitiveWithInfer):
    """Auto parallel virtual operator. Do nothing in forward, do Add in backward."""
    @prim_attr_register
    def __init__(self):
        """init"""

    def infer_shape(self, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        return x_dtype


class _VirtualDataset(PrimitiveWithInfer):
    """
    Auto parallel virtual dataset operator.

    It would insert Broadcast operator in forward computation and be deleted before backward computation.
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def infer_shape(self, *args):
        return args

    def infer_dtype(self, *args):
        return args


virtual_dataset = _VirtualDataset()


class _GetTensorSlice(PrimitiveWithInfer):
    """
    Gets tensor slice by device matrix and tensor map.

    Args:
        dev_mat (tuple): The device matrix of the slice tensor.
        tensor_map (tuple): The tensor map of the slice tensor.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ChunkTensor"""

    def infer_value(self, x, dev_mat, tensor_map):
        from mindspore.parallel._tensor import _load_tensor
        validator.check_value_type("dev_mat", dev_mat, [tuple], self.name)
        validator.check_value_type("tensor_map", tensor_map, [tuple], self.name)
        return Tensor(_load_tensor(x, dev_mat, tensor_map))
