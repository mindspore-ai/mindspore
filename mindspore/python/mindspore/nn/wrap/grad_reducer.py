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
"""grad reducer cell for distributed training"""
from __future__ import absolute_import

from mindspore import context
from mindspore import log as logger
from mindspore.nn.cell import Cell
from mindspore.communication.management import GlobalComm, get_group_size
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.ops import functional as F, composite as C
from mindspore.ops.operations.comm_ops import AllReduce, AllGather
from mindspore.parallel._auto_parallel_context import auto_parallel_context
import mindspore.common.dtype as mstype
from mindspore.common.sparse_tensor import Tensor
from mindspore.common.api import jit


reduce_opt = C.MultitypeFuncGraph("reduce_opt")


def _init_allreduce_operators(length, split_indices, group=GlobalComm.WORLD_COMM_GROUP):
    """ initialize allreduce communication operators"""
    for indices in split_indices:
        if indices >= length:
            logger.warning(f"AllReduce's split index {indices} is greater than or equal to"
                           f"the total gradient's number of {length}")
    fusion_type = 2 ** 10
    split = 0
    fusion = ()
    for i in range(length):
        fusion = fusion + (fusion_type,)
        if split >= len(split_indices):
            continue
        if split_indices[split] <= i:
            fusion_type += 1
            split += 1

    index = tuple(range(1, length + 1))
    op_list = ()
    for i in range(length):
        op = AllReduce('sum', group)
        op_fusion_id = fusion[i]
        # When running in ge and enabled all_reduce_fusion_config, hccl will check the allreduce' fusion id to be -1
        if context.get_context("enable_ge") and context.get_auto_parallel_context("all_reduce_fusion_config"):
            op_fusion_id = -1
        op.add_prim_attr('fusion', op_fusion_id)
        op.add_prim_attr('index', index[i])
        op_list = op_list + (op,)
    return op_list


def _init_allreduce_operators_by_parameters(parameters, split_indices, group, fusion_type=1):
    """ initialize allreduce communication operators by parameters"""
    op_list = ()
    param_fusion = False
    last_comm_fusion = None
    first_parameter_flag = True
    index = 1
    for parameter in parameters:
        comm_fusion = parameter.comm_fusion
        if first_parameter_flag:
            last_comm_fusion = comm_fusion
            first_parameter_flag = False
        elif not param_fusion:
            if comm_fusion != last_comm_fusion:
                param_fusion = True
                last_comm_fusion = comm_fusion
        op = AllReduce('sum', group)
        op.add_prim_attr('fusion', comm_fusion)
        op.add_prim_attr('index', index)
        index += 1
        op_list = op_list + (op,)

    if not param_fusion:
        if split_indices and fusion_type == 1:
            op_list = _init_allreduce_operators(len(parameters), split_indices, group)
            param_fusion = True
        else:
            op_list = ()
    return op_list, param_fusion


@reduce_opt.register("Tensor", "Bool", "Function", "Function", "Bool", "Tensor")
def _tensors_allreduce(degree, mean, allgather, allreduce, allreduce_filter, grad):
    """
    Apply allreduce on gradient.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allreduce would apply.
        grad (Tensor): The gradient tensor before operation.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    if allreduce_filter:
        grad = allreduce(grad)
        if mean:
            grad = F.tensor_mul(grad, F.cast(degree, F.dtype(grad)))
        return grad
    return grad


@reduce_opt.register("Tensor", "Bool", "Bool", "Tensor")
def _tensors_allreduce_post(degree, mean, allreduce_filter, grad):
    """
    Apply allreduce on gradient in PyNative mode.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allreduce_filter (bool): When it is true, allreduce would apply.
        grad (Tensor): The gradient tensor before operation.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    if allreduce_filter:
        if mean:
            grad = F.tensor_mul(grad, F.cast(degree, F.dtype(grad)))
            return grad
    return grad


@reduce_opt.register("Tensor", "Bool", "Function", "Function", "Bool", "Tensor", "Bool")
def _tensors_allreduce_ps(degree, mean, allgather, allreduce, allreduce_filter, grad, ps_parameter):
    """
    Apply allreduce on gradient.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allreduce would apply.
        grad (Tensor): The gradient tensor before operation.
        ps_parameter (bool): Use parameter server or not.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    if ps_parameter:
        return grad

    if allreduce_filter:
        grad = allreduce(grad)
        if mean:
            grad = F.tensor_mul(grad, F.cast(degree, F.dtype(grad)))
        return grad
    return grad


@reduce_opt.register("Tensor", "Bool", "Function", "Function", "Bool", "RowTensor")
def _tensors_allreduce_with_sparse(degree, mean, allgather, allreduce, allreduce_filter, grad):
    """
    Apply allgather on gradient instead of allreduce for sparse feature.
    Allgather is a communication operation used for distributed deep learning.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allgather would apply.
        grad (tuple): The indices, gradient tensor and tensor_shape before operation.

    Returns:
        RowTensor, the gradient after operation.
    """
    if allreduce_filter:
        indices = allgather(grad.indices)
        dout = allgather(grad.values)
        if mean:
            dout = F.tensor_mul(dout, F.cast(degree, F.dtype(dout)))
        grad = RowTensorInner(indices, dout, grad.dense_shape)
    return grad


@reduce_opt.register("Tensor", "Bool", "Function", "Function", "Bool", "RowTensor", "Bool")
def _tensors_allreduce_with_sparse_ps(degree, mean, allgather, allreduce, allreduce_filter, grad, ps_parameter):
    """
    Apply allgather on gradient instead of allreduce for sparse feature.
    Allgather is a communication operation used for distributed deep learning.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allgather would apply.
        grad (tuple): The indices, gradient tensor and tensor_shape before operation.
        ps_parameter (bool): Use parameter server or not.

    Returns:
        RowTensor, the gradient after operation.
    """
    if ps_parameter:
        return grad

    if allreduce_filter:
        indices = allgather(grad.indices)
        dout = allgather(grad.values)
        if mean:
            dout = F.tensor_mul(dout, F.cast(degree, F.dtype(dout)))
        grad = RowTensorInner(indices, dout, grad.dense_shape)
    return grad


_get_datatype = C.MultitypeFuncGraph("_get_datatype")


@_get_datatype.register("Tensor")
def _tensors_get_datatype(grad):
    """
    Acquire gradient datatype.

    Args:
        grad (Tensor): The gradient tensor before operation.

    Returns:
        mstype, the datatype of gradient.
    """
    return F.dtype(grad)


@_get_datatype.register("RowTensor")
def _tensors_get_datatype_with_sparse(grad):
    """
    Acquire gradient datatype.

    Args:
        grad (RowTensor): The gradient before operation.

    Returns:
        mstype, the datatype of gradient.
    """
    return F.dtype(grad.values)


_cast_datatype = C.MultitypeFuncGraph("_cast_datatype")


@_cast_datatype.register("TypeType", "Tensor")
def _tensors_cast_datatype(datatype, grad):
    """
    Cast gradient to datatype.

    Args:
        datatype (mstype): the destination datatype of gradient.
        grad (Tensor): The gradient tensor before operation.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    return F.cast(grad, datatype)


@_cast_datatype.register("TypeType", "RowTensor")
def _tensors_cast_datatype_with_sparse(datatype, grad):
    """
    Cast gradient to datatype.

    Args:
        datatype (mstype): the destination datatype of gradient.
        grad (RowTensor): The gradient before operation.

    Returns:
        RowTensor, the gradient after operation.
    """
    dout = F.cast(grad.values, datatype)
    return RowTensorInner(grad.indices, dout, grad.dense_shape)


class DistributedGradReducer(Cell):
    """
    A distributed optimizer.

    Aggregate the gradients for all cards by using AllReduce in data parallel.

    Args:
        parameters (list): the parameters to be updated.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
                     When it is not specified, using the configuration `gradients_mean` in auto_parallel_context.
                     Default: ``None`` .
        degree (int): The mean coefficient. Usually it equals to device number. Default: ``None`` .
        fusion_type (int): The type of all reduce fusion. Default: ``1`` .
        group (str): The communication group to work on. Normally, the group should be created by create_group,
                     otherwise, using the default group. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

    Raises:
        ValueError: If degree is not an int or less than 0.

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
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindspore import Parameter, Tensor, ops, nn
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> ms.reset_auto_parallel_context()
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        >>>
        >>> class TrainingWrapper(nn.Cell):
        ...     def __init__(self, network, optimizer, sens=1.0):
        ...         super(TrainingWrapper, self).__init__(auto_prefix=False)
        ...         self.network = network
        ...         self.network.add_flags(defer_inline=True)
        ...         self.weights = optimizer.parameters
        ...         self.optimizer = optimizer
        ...         self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        ...         self.sens = sens
        ...         self.reducer_flag = False
        ...         self.grad_reducer = None
        ...         self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        ...         self.depend = ops.Depend()
        ...         if self.parallel_mode in [ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL]:
        ...             self.reducer_flag = True
        ...         if self.reducer_flag:
        ...             mean = context.get_auto_parallel_context("gradients_mean")
        ...             degree = context.get_auto_parallel_context("device_num")
        ...             self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        ...
        ...     def construct(self, *args):
        ...         weights = self.weights
        ...         loss = self.network(*args)
        ...         sens = F.fill(ops.DType()(loss), ops.Shape()(loss), self.sens)
        ...         grads = self.grad(self.network, weights)(*args, sens)
        ...         if self.reducer_flag:
        ...             # apply grad reducer on grads
        ...             grads = self.grad_reducer(grads)
        ...         return self.depend(loss, self.optimizer(grads))
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        >>>
        >>> size, in_features, out_features = 16, 16, 10
        >>> network = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> net_with_loss = nn.WithLossCell(network, loss)
        >>> optimizer = nn.Momentum(net_with_loss.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> train_cell = TrainingWrapper(net_with_loss, optimizer)
        >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
        >>> label = Tensor(np.zeros([size, out_features]).astype(np.float32))
        >>> grads = train_cell(inputs, label)
        >>> print(grads)
        256.0
    """

    def __init__(self, parameters, mean=None, degree=None, fusion_type=1, group=GlobalComm.WORLD_COMM_GROUP):
        super(DistributedGradReducer, self).__init__(auto_prefix=False)
        self._check_parallel_mode()
        self.map_ = C.Map()
        self.mean = mean
        if mean is None:
            self.mean = auto_parallel_context().get_gradients_mean()
        if degree is None:
            self.degree = get_group_size()
        else:
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError("For 'DistributedGradReducer', "
                                 "parameter 'degree' in DistributedGradReducer "
                                 "should large than 0 and be int, degree: {}.".format(degree))
            self.degree = degree
        self.degree = Tensor(1.0 / self.degree, mstype.float32)

        self.allreduce_filter = tuple((x.layerwise_parallel is False) and (x.is_in_shard is False) for x in parameters)
        is_parallel_optimizer = context.get_auto_parallel_context("enable_parallel_optimizer")
        split_indices = auto_parallel_context().get_all_reduce_fusion_split_indices()
        if is_parallel_optimizer and split_indices:
            self.split_fusion = True
            self.op_list = _init_allreduce_operators(len(parameters), split_indices, group)
        else:
            self.split_fusion = True
            self.op_list, param_fusion = _init_allreduce_operators_by_parameters(parameters, split_indices, group,
                                                                                 fusion_type)
            if not param_fusion:
                self.split_fusion = False
                self.allreduce = AllReduce('sum', group).add_prim_attr('fusion', fusion_type)
        self.allgather = AllGather(group)
        ps_filter = lambda x: x.is_param_ps
        self.ps_parameters = tuple(ps_filter(x) for x in parameters)
        self.enable_parameter_server = any(self.ps_parameters)
        self.mode = context.get_context("mode")
        self.enable_tuple_broaden = True

    @jit
    def construct(self, grads):
        """
        Under certain circumstances, the data precision of grads could be mixed with float16 and float32. Thus, the
        result of AllReduce is unreliable. To solve the problem, grads must be cast to float32 before AllReduce,
        and cast back after the operation.

        Args:
            grads (Union[Tensor, tuple[Tensor]]): The gradient tensor or tuple before operation.

        Returns:
            new_grads (Union[Tensor, tuple[Tensor]]), the gradient tensor or tuple after operation.
        """
        datatypes = self.map_(F.partial(_get_datatype), grads)
        grads = self.map_(F.partial(_cast_datatype, mstype.float32), grads)

        if self.split_fusion:
            if self.enable_parameter_server:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather),
                                     self.op_list, self.allreduce_filter, grads, self.ps_parameters)
            else:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather),
                                     self.op_list, self.allreduce_filter, grads)
        else:
            if self.enable_parameter_server:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather,
                                               self.allreduce), self.allreduce_filter, grads, self.ps_parameters)
            else:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather,
                                               self.allreduce), self.allreduce_filter, grads)
        new_grad = self.map_(F.partial(_cast_datatype), datatypes, new_grad)
        return new_grad

    def _check_parallel_mode(self):
        """check parallel mode"""
        parallel_mode = context.get_auto_parallel_context('parallel_mode')
        if context.get_context('mode') == context.GRAPH_MODE and parallel_mode in (
                context.ParallelMode.SEMI_AUTO_PARALLEL, context.ParallelMode.AUTO_PARALLEL):
            raise RuntimeError("{} can not use DistributedGradReducer in graph mode".format(parallel_mode))
