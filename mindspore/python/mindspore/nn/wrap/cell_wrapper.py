
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
"""Cell_wrapper."""
from __future__ import absolute_import
from __future__ import division

from types import FunctionType, MethodType

from mindspore import log as logger
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean,\
    _get_parallel_mode, _get_enable_parallel_optimizer, _is_pynative_parallel
from mindspore.context import ParallelMode
from mindspore._checkparam import Validator as validator
from mindspore import ops, nn
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import _VirtualDataset
from mindspore.nn.cell import Cell
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

_get_datatype = C.MultitypeFuncGraph("_get_datatype")


@_get_datatype.register("Tensor")
def _tensors_get_datatype(param):
    """
    Acquire parameter datatype.

    Args:
        param (Tensor): The parameter before operation.

    Returns:
        mstype, the datatype of parameter.
    """
    return F.dtype(param)


_cast_datatype = C.MultitypeFuncGraph("_cast_datatype")


@_cast_datatype.register("TypeType", "Tensor")
def _tensors_cast_datatype(datatype, param):
    """
    Cast gradient to datatype.

    Args:
        datatype (mstype): the destination datatype of parameter.
        param (Tensor): The parameter before operation.

    Returns:
        Tensor, the parameter after operation.
    """
    return F.cast(param, datatype)


class WithLossCell(Cell):
    r"""
    Cell with loss function.

    Wraps the network with loss function. This Cell accepts data and label as inputs and
    the computed loss will be returned.

    Args:
        backbone (Cell): The backbone network to wrap.
        loss_fn (Cell): The loss function used to compute loss.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If dtype of `data` or `label` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        >>> net_with_criterion = nn.WithLossCell(net, loss_fn)
        >>>
        >>> batch_size = 2
        >>> data = Tensor(np.ones([batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
        >>> label = Tensor(np.ones([batch_size, 10]).astype(np.float32))
        >>>
        >>> output_data = net_with_criterion(data, label)
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        if isinstance(backbone, Cell) and backbone.jit_config_dict:
            self._jit_config_dict = backbone.jit_config_dict

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone


class WithGradCell(Cell):
    r"""
    Cell that returns the gradients.

    Wraps the network with backward cell to compute gradients. A network with a loss function is necessary
    as argument. If loss function in None, the network must be a wrapper of network and loss function. This
    Cell accepts '\*inputs' as inputs and returns gradients for each trainable parameter.

    Note:
        Run in PyNative mode.

    Args:
        network (Cell): The target network to wrap. The network only supports single output.
        loss_fn (Cell): Primitive loss function used to compute gradients. Default: None.
        sens (Union[None, Tensor, Scalar, Tuple ...]): The sensitive for backpropagation, the type and shape
            must be same as the `network` output. If None, we will fill one to a same type shape of
            output value. Default: None.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        list, a list of Tensors with identical shapes as trainable weights.

    Raises:
        TypeError: If `sens` is not one of None, Tensor, Scalar or Tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> grad_net = nn.WithGradCell(net, loss_fn)
        >>>
        >>> # For a network wrapped with loss function
        >>> net = Net()
        >>> net_with_criterion = nn.WithLossCell(net, loss_fn)
        >>> grad_net = nn.WithGradCell(net_with_criterion)
    """

    def __init__(self, network, loss_fn=None, sens=None):
        super(WithGradCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation(
            get_by_list=True, sens_param=(sens is not None))
        self.sens = sens
        if loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLossCell(self.network, self.loss_fn)
        self.network_with_loss.set_train()
        if isinstance(network, Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

    def construct(self, *inputs):
        weights = self.weights
        if self.sens is None:
            grads = self.grad(self.network_with_loss, weights)(*inputs)
        else:
            grads = self.grad(self.network_with_loss,
                              weights)(*inputs, self.sens)
        return grads


class ForwardValueAndGrad(Cell):
    r"""
    Encapsulate training network.

    Including the network and a gradient function. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the gradient function to calculating gradient.

    Args:
        network (Cell): The training network.
        weights (ParameterTuple): The parameters of the training network that need to calculate the gradient.
            Default: None.
        get_all (bool): If True, get all the gradients with respect to inputs. Default: False.
        get_by_list (bool): If True, get all the gradients with respect to Parameter variables.
            If get_all and get_by_list are both False, get the gradient with respect to first input.
            If get_all and get_by_list are both True, get the gradients with respect to inputs and Parameter variables
            at the same time in the form of ((gradients with respect to inputs),
            (gradients with respect to parameters)). Default: False.
        sens_param (bool): Whether to append sensitivity (gradient with respect to output) as input.
            If sens_param is False, a 'ones_like(outputs)' sensitivity will be attached automatically.
            Default: False.
            If the sens_param is True, a sensitivity (gradient with respect to output) needs to be transferred through
            the input parameter.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor...)) - Tuple of inputs with shape :math:`(N, \ldots)`.
        - **(sens)** - A sensitivity (gradient with respect to output) as the input of backpropagation.
          If network has single output, the sens is a tensor.
          If network has multiple outputs, the sens is the tuple(tensor).

    Outputs:
        - **forward value** - The result of network forward running.
        - **gradients** (tuple(tensor)) - The gradients of network parameters and inputs.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn, common, ops, ParameterTuple, Parameter
        >>>
        >>> class Net(nn.Cell):
        ...    def __init__(self):
        ...        super(Net, self).__init__()
        ...        self.weight = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="weight")
        ...        self.matmul = ops.MatMul()
        ...
        ...    def construct(self, x):
        ...        out = self.matmul(x, self.weight)
        ...        return out
        ...
        >>> net = Net()
        >>> criterion = nn.SoftmaxCrossEntropyWithLogits()
        >>> net_with_criterion = nn.WithLossCell(net, criterion)
        >>> weight = ParameterTuple(net.trainable_params())
        >>> train_network = nn.ForwardValueAndGrad(net_with_criterion, weights=weight, get_all=True, get_by_list=True)
        >>> inputs = Tensor(np.ones([1, 2]).astype(np.float32))
        >>> labels = Tensor(np.ones([1, 2]).astype(np.float32))
        >>> result = train_network(inputs, labels)
        >>> print(result)
         (Tensor(shape=[1], dtype=Float32, value= [ 1.38629436e+00]), ((Tensor(shape=[1, 2], dtype=Float32, value=
        [[ -1.00000000e+00,  -1.00000000e+00]]), Tensor(shape=[1, 2], dtype=Float32, value=
        [[ 0.00000000e+00,  0.00000000e+00]])), (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ -5.00000000e-01,  -5.00000000e-01],
         [ -5.00000000e-01,  -5.00000000e-01]]),)))
    """

    def __init__(self, network, weights=None, get_all=False, get_by_list=False, sens_param=False):
        super(ForwardValueAndGrad, self).__init__(auto_prefix=False)
        if not isinstance(network, (Cell, FunctionType, MethodType)):
            raise TypeError(f"For 'ForwardValueAndGrad', "
                            f"the argument 'network' must be cell, function type or method type, "
                            f"but got '{type(network)}'")
        if not isinstance(get_all, bool):
            raise TypeError(f"For 'ForwardValueAndGrad', "
                            f"the type of 'get_all' must be bool, but got '{type(get_all)}'")
        if not isinstance(get_by_list, bool):
            raise TypeError(f"For 'ForwardValueAndGrad', "
                            f"the type of 'get_by_list' must be bool, but got '{type(get_by_list)}'")
        if get_by_list and not isinstance(weights, (ParameterTuple, tuple, list)):
            raise TypeError(f"For 'ForwardValueAndGrad', "
                            f"when 'get_by_list' is set to True, the argument 'weights' must be "
                            f"Parameters array, but got '{type(weights)}'")
        self.network = network
        if isinstance(network, Cell):
            self.network.set_grad()
        self.weights = weights
        self.get_all = get_all
        self.get_by_list = get_by_list
        self.sens_param = sens_param
        self.grad = C.GradOperation(
            get_all=self.get_all, get_by_list=self.get_by_list, sens_param=self.sens_param)
        if isinstance(network, Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

    def construct(self, *inputs):
        grad_inputs = inputs
        if self.sens_param:
            inputs = inputs[:-1]
        loss = self.network(*inputs)
        if self.get_by_list:
            grads = self.grad(self.network, self.weights)(*grad_inputs)
        else:
            grads = self.grad(self.network)(*grad_inputs)
        return loss, grads


class TrainOneStepCell(Cell):
    r"""
    Network training package class.

    Wraps the `network` with the `optimizer`. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Union[Cell]): Optimizer for updating the network parameters.
        sens (numbers.Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If `sens` is not a numbers.Number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> #1) Using the WithLossCell provided by MindSpore
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
        >>>
        >>> #2) Using user-defined WithLossCell
        >>> class MyWithLossCell(Cell):
        ...    def __init__(self, backbone, loss_fn):
        ...        super(MyWithLossCell, self).__init__(auto_prefix=False)
        ...        self._backbone = backbone
        ...        self._loss_fn = loss_fn
        ...
        ...    def construct(self, x, y, label):
        ...        out = self._backbone(x, y)
        ...        return self._loss_fn(out, label)
        ...
        ...    @property
        ...    def backbone_network(self):
        ...        return self._backbone
        ...
        >>> loss_net = MyWithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = self.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL) or \
            _is_pynative_parallel()
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            from mindspore.communication.management import GlobalComm
            group = GlobalComm.WORLD_COMM_GROUP
            if isinstance(self.optimizer, (nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell)):
                from mindspore.communication.management import get_group_size, create_group, get_rank
                group_number = get_group_size() // 8
                self.degree = int(self.degree / group_number)
                group_list = [list(range(x * self.degree, (x + 1) * self.degree))
                              for x in range(group_number)]
                current_index = get_rank() // 8
                server_group_name = "allreduce_" + str(current_index)
                create_group(server_group_name, group_list[current_index])
                group = server_group_name
            self.grad_reducer = DistributedGradReducer(
                self.weights, self.mean, self.degree, group=group)
        if isinstance(network, Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

    def identity(self, x):
        return x


class GetNextSingleOp(Cell):
    """
    Cell to run for getting the next operation.

    For detailed information, refer to :class:`mindspore.ops.GetNext`.

    Args:
        dataset_types (list[:class:`mindspore.dtype`]): The types of dataset.
        dataset_shapes (list[tuple[int]]): The shapes of dataset.
        queue_name (str): Queue name to fetch the data.

    Outputs:
        tuple[Tensor], the data get from Dataset.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops, nn
        >>> from mindspore import dataset as ds
        >>> from mindspore.common import dtype as mstype
        >>>
        >>> data_path =  "/path/to/MNIST_Data/train/"
        >>> train_dataset = ds.MnistDataset(data_path, num_samples=10)
        >>> dataset_helper = mindspore.DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>> dataset = dataset_helper.iter.dataset
        >>> dataset_types, dataset_shapes = dataset_helper.types_shapes()
        >>> queue_name = dataset.__transfer_dataset__.queue_name
        >>> get_next_single_op_net = nn.GetNextSingleOp(dataset_types, dataset_shapes, queue_name)
        >>> data, label = get_next_single_op_net()
        >>> relu = ops.ReLU()
        >>> result = relu(data.astype(mstype.float32))
        >>> print(result.shape)
        (28, 28, 1)
    """

    def __init__(self, dataset_types, dataset_shapes, queue_name):
        super(GetNextSingleOp, self).__init__()
        self.get_next = P.GetNext(
            dataset_types, dataset_shapes, len(dataset_types), queue_name)

    def construct(self):
        return self.get_next()


class _VirtualDatasetCell(Cell):
    """
    Wrap the network with virtual dataset to convert data parallel layout to model parallel layout.

    _VirtualDataset is a virtual Primitive, it does not exist in the final executing graph. Inputs and outputs
    of _VirtualDataset are distributed in data parallel pattern, tensor redistribution Primitives is inserted
    dynamically during the graph compile process.

    Note:
        Only used in semi auto parallel and auto parallel mode.

    Args:
        backbone (Cell): The target network to wrap.

    Examples:
        >>> net = Net()
        >>> net = _VirtualDatasetCell(net)
    """

    def __init__(self, backbone):
        super(_VirtualDatasetCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()
        if isinstance(backbone, Cell) and backbone.jit_config_dict:
            self._jit_config_dict = backbone.jit_config_dict

    def construct(self, *inputs):
        output = self._virtual_dataset(*inputs)
        return self._backbone(*output)


class _MicroBatch(Cell):
    """
    transform mini-batch to micro-batch in pipeline parallel.

    Args:
       params (micro_size): The number of micro-batch.
    """

    def __init__(self, micro_size):
        super(_MicroBatch, self).__init__()
        self.shape = P.Shape()
        self.micro_size = micro_size
        self.strided_slice = P.StridedSlice()

    def construct(self, i, *inputs):
        """construct for _MicroBatch."""
        micro_inputs = ()
        for each_input in inputs:
            input_shape = self.shape(each_input)
            micro_batch_begin = i * input_shape[0] // self.micro_size
            micro_batch_end = (i + 1) * input_shape[0] // self.micro_size
            strided_slice_begin = (micro_batch_begin,)
            strided_slice_strides = (1,)
            for _ in range(len(input_shape) - 1):
                strided_slice_begin += (0,)
                strided_slice_strides += (1,)
            strided_slice_end = (micro_batch_end,)
            strided_slice_end += input_shape[1:]
            micro_input = self.strided_slice(
                each_input, strided_slice_begin, strided_slice_end, strided_slice_strides)
            micro_inputs += (micro_input,)
        return micro_inputs


class MicroBatchInterleaved(Cell):
    """
    This function splits the input at the 0th into interleave_num pieces and then performs
    the computation of the wrapped cell. Application scenario: When there is model parallelism in semi-automatic mode
    and network, if the first slice data is calculating forward, the second slice data will execute the
    communication operators at the same time, to achieve the performance acceleration of communication and computing
    concurrency.

    Note:
        The output of the input network must be a single tensor.

    Args:
        network (Cell): The target network to wrap.
        interleave_num (int, optional): split num of batch size. Default: 2.

    Inputs:
        tuple[Tensor]. It's the same with the input of the `network` .

    Outputs:
        Tensor. The output of the input `network` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = Net()
        >>> net = MicroBatchInterleaved(net, 2)
    """

    def __init__(self, network, interleave_num=2):
        super(MicroBatchInterleaved, self).__init__(auto_prefix=False)
        if not isinstance(interleave_num, int):
            raise TypeError("For 'MicroBatchInterleaved', the argument 'interleave_num' must be integer, "
                            "but got the type : {}.".format(type(interleave_num)))
        if interleave_num <= 0:
            raise ValueError("For 'MicroBatchInterleaved', the argument 'interleave_num' must be large than 0, "
                             "but got {}.".format(interleave_num))
        self.network = network
        self.interleave_num = interleave_num
        self.interleave_inputs = nn.CellList()
        self.add = P.Add().add_prim_attr("micro_interleaved_add_flag", True)
        for _ in range(interleave_num):
            interleave_data = _MicroBatch(interleave_num)
            interleave_data.strided_slice.add_prim_attr(
                "strided_slice_flag", True)
            interleave_data.strided_slice.add_prim_attr(
                "interleave_num", interleave_num)
            self.interleave_inputs.append(interleave_data)
        if isinstance(network, Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

    def construct(self, *inputs):
        output = 0.0
        for i in range(self.interleave_num):
            interleave_input = self.interleave_inputs[i](i, *inputs)
            output = self.add(output, self.network(*interleave_input))
        return output


class PipelineCell(Cell):
    """
    Wrap the network with Micro Batch.

    Note:
        micro_size must be greater or equal to pipeline stages.

    Args:
        network (Cell): The target network to wrap.
        micro_size (int): MicroBatch size.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = Net()
        >>> net = PipelineCell(net, 4)
    """

    def __init__(self, network, micro_size):
        super(PipelineCell, self).__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("pipeline_end", i)
            self.add_list.append(self.add)
        if isinstance(network, Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

    def construct(self, *inputs):
        ret = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output = self.network(*micro_input)
            if ret is not None:
                ret = self.add_list[i](ret, output)
            else:
                ret = output
        return ret


def _pipeline_clear_grad(accu_grad, grad):
    accu_grad = F.depend(accu_grad, grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    return F.assign(accu_grad, zeros)


class _TrainPipelineAccuStepCell(TrainOneStepCell):
    """
    Wraps the network with an optimizer in pipeline mode.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(_TrainPipelineAccuStepCell, self).__init__(
            network, optimizer, sens)
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.hyper_map = ops.HyperMap()
        self.opt_shard = _get_enable_parallel_optimizer()
        if isinstance(network, Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        accu_grads = ops.depend(self.accu_grads, grads)
        if self.opt_shard:
            succ = self.optimizer(grads)
        else:
            succ = self.optimizer(accu_grads)
        loss = ops.depend(loss, succ)
        clear = self.hyper_map(_pipeline_clear_grad, accu_grads, grads)
        loss = ops.depend(loss, clear)
        return loss


class VirtualDatasetCellTriple(Cell):
    """
    Wrap the network with virtual dataset to convert data parallel layout to model parallel layout.

    VirtualDatasetCellTriple is a virtual Primitive, it does not exist in the final executing graph. Inputs and outputs
    of VirtualDatasetCellTriple are distributed in data parallel pattern, tensor redistribution Primitives is inserted
    dynamically during the graph compile process.

    Note:
        Only used in semi auto parallel and auto parallel mode. There are three inputs, as contrary to two inputs in
        _VirtualDatasetCell.

    Args:
        backbone (Cell): The target network to wrap.

    Examples:
        >>> net = Net()
        >>> net = VirtualDatasetCellTriple(net)
    """

    def __init__(self, backbone):
        super(VirtualDatasetCellTriple, self).__init__(auto_prefix=False)
        logger.warning(
            "WARN_DEPRECATED: The usage of VirtualDatasetCellTriple is deprecated.")
        self._backbone = backbone
        if isinstance(backbone, Cell) and backbone.jit_config_dict:
            self._jit_config_dict = backbone.jit_config_dict

    def construct(self, a, b, c):
        return self._backbone(a, b, c)


class WithEvalCell(Cell):
    r"""
    Wraps the forward network with the loss function.

    It returns loss, forward output and label to calculate the metrics.

    Args:
        network (Cell): The forward network.
        loss_fn (Cell): The loss function.
        add_cast_fp32 (bool): Whether to adjust the data type to float32. Default: False.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tuple(Tensor), containing a scalar loss Tensor, a network output Tensor of shape :math:`(N, \ldots)`
        and a label Tensor of shape :math:`(N, \ldots)`.

    Raises:
        TypeError: If `add_cast_fp32` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # Forward network without loss function
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> eval_net = nn.WithEvalCell(net, loss_fn)
    """

    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = validator.check_value_type(
            "add_cast_fp32", add_cast_fp32, [bool], self.cls_name)
        if isinstance(network, Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

    def construct(self, data, label):
        outputs = self._network(data)
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        loss = self._loss_fn(outputs, label)
        return loss, outputs, label


class ParameterUpdate(Cell):
    """
    Cell that updates parameter.

    With this Cell, one can manually update `param` with the input `Tensor`.

    Args:
        param (Parameter): The parameter to be updated manually.

    Inputs:
        - **x** (Tensor) - A tensor whose shape and type are the same with `param`.

    Outputs:
        Tensor, the input `x`.

    Raises:
        KeyError: If parameter with the specified name does not exist.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import nn, Tensor
        >>> network = nn.Dense(3, 4)
        >>> param = network.parameters_dict()['weight']
        >>> update = nn.ParameterUpdate(param)
        >>> update.phase = "update_param"
        >>> weight = Tensor(np.arange(12).reshape((4, 3)), mindspore.float32)
        >>> output = update(weight)
        >>> print(output)
        [[ 0.  1.  2.]
         [ 3.  4.  5.]
         [ 6.  7.  8.]
         [ 9. 10. 11.]]
    """

    def __init__(self, param):
        super(ParameterUpdate, self).__init__(auto_prefix=False)
        if not isinstance(param, Parameter):
            raise TypeError(
                "For 'ParameterUpdate', 'param' must be 'Parameter', but got {}.".format(type(param)))
        self._param = param

    def construct(self, x):
        F.assign(self._param, x)
        return x


class _BroadCastCell(Cell):
    """
    Broadcast the parameters from device 0 to other devices.

    Args:
       params (list): The parameters of Net.
    """

    def __init__(self, params):
        super(_BroadCastCell, self).__init__()
        from mindspore.communication.management import get_group_size, create_group
        from mindspore import context
        self.map_ = C.Map()
        self.params = tuple(params)
        if context.get_context("device_target") == "Ascend" and context.get_context("mode") != context.PYNATIVE_MODE:
            rank_list = [id for id in range(0, get_group_size())]
            create_group("BroadcastWorldGroup", rank_list)
            self.broadcast = P.Broadcast(0, group="BroadcastWorldGroup")
        else:
            self.broadcast = P.Broadcast(0)
        self.add_flags(skip_auto_parallel_compile=True)

    def construct(self):
        datatypes = self.map_(F.partial(_get_datatype), self.params)
        params = self.map_(
            F.partial(_cast_datatype, mstype.float32), self.params)
        params = self.broadcast(params)
        new_params = self.map_(F.partial(_cast_datatype), datatypes, params)
        return new_params
