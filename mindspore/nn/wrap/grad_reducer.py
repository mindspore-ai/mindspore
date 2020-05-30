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
from mindspore.nn.cell import Cell
from mindspore.communication.management import GlobalComm, get_group_size
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.ops.operations.comm_ops import AllReduce, ReduceOp
import mindspore.common.dtype as mstype

reduce_opt = C.MultitypeFuncGraph("reduce_opt")

_all_reduce = AllReduce()


def _init_optimizer_allreduce():
    global _all_reduce
    _all_reduce = AllReduce(ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)
    _all_reduce.add_prim_attr('fusion', 1)


@reduce_opt.register("Function", "Number", "Bool", "Tensor")
def _tensors_allreduce_mean(mul, degree, allreduce_filter, grad):
    """
    Apply mean and allreduce on gradient. Allreduce is a communication operation used for distributed deep learning.

    Args:
        mul (Primitive): Div operation.
        degree (int): The mean coefficient.
        allreduce_filter (bool): When it is true, allreduce would apply.
        grad (Tensor): The gradient tensor before operation.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    if allreduce_filter:
        degree = F.scalar_cast(degree, F.dtype(grad))
        grad = _all_reduce(grad)
        cast_op = P.Cast()
        return mul(grad, cast_op(F.scalar_to_array(1.0/degree), F.dtype(grad)))
    return grad


@reduce_opt.register("Function", "Number", "Bool", "Tuple")
def _tensors_allreduce_mean_with_sparse(mul, degree, allreduce_filter, grad):
    """
    Apply mean and allgather on gradient instead of allreduce for sparse feature.
    Allgather is a communication operation used for distributed deep learning.

    Args:
        mul (Primitive): Div operation.
        degree (int): The mean coefficient.
        allreduce_filter (bool): When it is true, allgather would apply.
        grad (Tuple): The indices, gradient tensor and tensor_shape before operation.

    Returns:
        Tuple, include indices, the gradient tensor and tensor_shape after operation.
    """
    if allreduce_filter:
        indices = _all_gather(grad[0])
        degree = F.scalar_cast(degree, F.dtype(grad[1]))
        dout = _all_gather(grad[1])
        cast_op = P.Cast()
        dout = mul(dout, cast_op(F.scalar_to_array(1.0/degree), F.dtype(dout)))
        grad = (indices, dout, dout[2])
    return grad


@reduce_opt.register("Bool", "Tensor")
def _tensors_allreduce(allreduce_filter, grad):
    """
    Apply allreduce on gradient.

    Args:
        allreduce_filter (bool): When it is true, allreduce would apply.
        grad (Tensor): The gradient tensor before operation.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    if allreduce_filter:
        return _all_reduce(grad)
    return grad


@reduce_opt.register("Bool", "Tuple")
def _tensors_allreduce_with_sparse(allreduce_filter, grad):
    """
    Apply mean and allgather on gradient instead of allreduce for sparse feature.
    Allgather is a communication operation used for distributed deep learning.

    Args:
        allreduce_filter (bool): When it is true, allgather would apply.
        grad (Tuple): The indices, gradient tensor and tensor_shape before operation.

    Returns:
        Tuple, include indices, the gradient tensor and tensor_shape after operation.
    """
    if allreduce_filter:
        indices = _all_gather(grad[0])
        dout = _all_gather(grad[1])
        grad = (indices, dout, dout[2])
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


class DistributedGradReducer(Cell):
    """
    A distributed optimizer.

    Constructs a gradient reducer Cell, which applies communication and average operations on
    single-process gradient values.

    Args:
        parameters (list): the parameters to be updated.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients. Default: False.
        degree (int): The mean coefficient. Usually it equals to device number. Default: None.

    Raises:
        ValueError: If degree is not a int or less than 0.

    Examples:
        >>> from mindspore.communication import init, get_group_size
        >>> from mindspore.ops import composite as C
        >>> from mindspore.ops import operations as P
        >>> from mindspore.ops import functional as F
        >>> from mindspore import context
        >>> from mindspore import nn
        >>> from mindspore import ParallelMode, ParameterTuple
        >>>
        >>> device_id = int(os.environ["DEVICE_ID"])
        >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True,
        >>>                     device_id=int(device_id))
        >>> init()
        >>> context.reset_auto_parallel_context()
        >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)
        >>>
        >>>
        >>> class TrainingWrapper(nn.Cell):
        >>>     def __init__(self, network, optimizer, sens=1.0):
        >>>         super(TrainingWrapper, self).__init__(auto_prefix=False)
        >>>         self.network = network
        >>>         self.network.add_flags(defer_inline=True)
        >>>         self.weights = optimizer.parameters
        >>>         self.optimizer = optimizer
        >>>         self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        >>>         self.sens = sens
        >>>         self.reducer_flag = False
        >>>         self.grad_reducer = None
        >>>         self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        >>>         if self.parallel_mode in [ParallelMode.DATA_PARALLEL,
        >>>                                            ParallelMode.HYBRID_PARALLEL]:
        >>>             self.reducer_flag = True
        >>>         if self.reducer_flag:
        >>>             mean = context.get_auto_parallel_context("mirror_mean")
        >>>             if mean.get_device_num_is_set():
        >>>                 degree = context.get_auto_parallel_context("device_num")
        >>>             else:
        >>>                 degree = get_group_size()
        >>>             self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        >>>
        >>>     def construct(self, *args):
        >>>         weights = self.weights
        >>>         loss = self.network(*args)
        >>>         sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        >>>         grads = self.grad(self.network, weights)(*args, sens)
        >>>         if self.reducer_flag:
        >>>             # apply grad reducer on grads
        >>>             grads = self.grad_reducer(grads)
        >>>         return F.depend(loss, self.optimizer(grads))
        >>>
        >>> network = Net()
        >>> optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> train_cell = TrainingWrapper(network, optimizer)
        >>> inputs = Tensor(np.ones([16, 16]).astype(np.float32))
        >>> label = Tensor(np.zeros([16, 16]).astype(np.float32))
        >>> grads = train_cell(inputs, label)
    """

    def __init__(self, parameters, mean=True, degree=None):
        super(DistributedGradReducer, self).__init__(auto_prefix=False)
        self.hyper_map = C.HyperMap()
        self.mul = P.Mul()
        if degree is None:
            self.degree = get_group_size()
        else:
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError("Parameter 'degree' in DistributedGradReducer should large than 0 and be int")
            self.degree = degree
        self.mean = mean
        self.allreduce_filter = tuple(x.layerwise_parallel is False for x in parameters)
        _init_optimizer_allreduce()

    def construct(self, grads):
        # In some circumstances, the data precision of grads could be mixed with float16 and float32. Thus, the
        # result of AllReduce is unreliable. To solve the problem, grads should be cast to float32 before AllReduce,
        # and cast back after the operation.
        datatypes = self.hyper_map(F.partial(_get_datatype), grads)
        grads = self.hyper_map(F.partial(_cast_datatype, mstype.float32), grads)

        if self.mean:
            new_grad = self.hyper_map(F.partial(reduce_opt, self.mul, self.degree), self.allreduce_filter, grads)
        else:
            new_grad = self.hyper_map(F.partial(reduce_opt), self.allreduce_filter, grads)

        new_grad = self.hyper_map(F.partial(_cast_datatype), datatypes, new_grad)
        return new_grad
