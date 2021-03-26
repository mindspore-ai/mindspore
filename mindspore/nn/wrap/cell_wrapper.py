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
from types import FunctionType, MethodType

from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore._checkparam import Validator as validator
from ...common import dtype as mstype
from ...common.parameter import Parameter, ParameterTuple
from ...ops import composite as C
from ...ops import functional as F
from ...ops import operations as P
from ...ops.operations.comm_ops import _VirtualDataset
from ..cell import Cell
from .grad_reducer import DistributedGradReducer

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
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar tensor with shape :math:`()`.

    Raises:
        TypeError: If dtype of `data` or `label` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

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

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        """
        Returns the backbone network.

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
        ``Ascend`` ``GPU``

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
        self.grad = C.GradOperation(get_by_list=True, sens_param=(sens is not None))
        self.sens = sens
        if loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLossCell(self.network, self.loss_fn)
        self.network_with_loss.set_train()

    def construct(self, *inputs):
        weights = self.weights
        if self.sens is None:
            grads = self.grad(self.network_with_loss, weights)(*inputs)
        else:
            grads = self.grad(self.network_with_loss, weights)(*inputs, self.sens)
        return grads


class ForwardValueAndGrad(Cell):
    r"""
    Network training package class.

    Including the network and a gradient function. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the gradient function to calculating gradient.

    Args:
        network (Cell): The training network.
        weights (ParameterTuple): The parameters of the training network that need to calculate the gradient.
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
        >>> class Net(nn.Cell):
        ...    def __init__(self):
        ...        super(Net, self).__init__()
        ...        self.weight = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="weight")
        ...        self.matmul = P.MatMul()
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
        >>> labels = Tensor(np.zeros([1, 2]).astype(np.float32))
        >>> result = train_network(inputs, labels)
        >>> print(result)
        (Tensor(shape=[1], dtype=Float32, value=[0.00000000e+00]), ((Tensor(shape=[1, 2], dtype=Float32, value=
        [[1.00000000e+00, 1.00000000e+00]]), Tensor(shape=[1, 2], dtype=Float32, value=
        [[0.00000000e+00, 0.00000000e+00]])), (Tensor(shape=[2, 2], dtype=Float32, value=
        [[5.00000000e-01, 5.00000000e-01],
         [5.00000000e-01, 5.00000000e-01]]),)))
    """

    def __init__(self, network, weights=None, get_all=False, get_by_list=False, sens_param=False):
        super(ForwardValueAndGrad, self).__init__(auto_prefix=False)
        if not isinstance(network, (Cell, FunctionType, MethodType)):
            raise TypeError(f"The type of training network should be cell, function type or method type, "
                            f"but got '{type(network)}'")
        if not isinstance(get_all, bool):
            raise TypeError(f"The type of get_all should be bool, but got '{type(get_all)}'")
        if not isinstance(get_by_list, bool):
            raise TypeError(f"The type of get_by_list should be bool, but got '{type(get_by_list)}'")
        if get_by_list and not isinstance(weights, ParameterTuple):
            raise TypeError(f"When get_by_list is set to True, the parameters of training network should be "
                            f"ParameterTuple type, but got '{type(weights)}'")
        self.network = network
        if isinstance(network, Cell):
            self.network.set_grad()
        self.weights = weights
        self.get_all = get_all
        self.get_by_list = get_by_list
        self.sens_param = sens_param
        self.grad = C.GradOperation(get_all=self.get_all, get_by_list=self.get_by_list, sens_param=self.sens_param)

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

    Wraps the network with an optimizer. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Raises:
        TypeError: If `sens` is not a number.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> #1) Using the WithLossCell existing provide
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
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss


class GetNextSingleOp(Cell):
    """
    Cell to run for getting the next operation.

    Args:
        dataset_types (list[:class:`mindspore.dtype`]): The types of dataset.
        dataset_shapes (list[tuple[int]]): The shapes of dataset.
        queue_name (str): Queue name to fetch the data.

    For detailed information, refer to `ops.operations.GetNext`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> train_dataset = create_custom_dataset()
        >>> dataset_helper = mindspore.DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>> dataset = dataset_helper.iter.dataset
        >>> dataset_types, dataset_shapes = dataset_helper.types_shapes()
        >>> queue_name = dataset.__transfer_dataset__.queue_name
        >>> get_next_single_op_net = nn.GetNextSingleOp(dataset_types, dataset_shapes, queue_name)
        >>> data, label = get_next_single_op_net()
        >>> relu = P.ReLU()
        >>> result = relu(data).asnumpy()
        >>> print(result.shape)
        (32, 1, 32, 32)
    """

    def __init__(self, dataset_types, dataset_shapes, queue_name):
        super(GetNextSingleOp, self).__init__()
        self.get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)

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

    def construct(self, *inputs):
        output = self._virtual_dataset(*inputs)
        return self._backbone(*output)


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
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, a, b, c):
        a_, b_, c_ = self._virtual_dataset(a, b, c)
        return self._backbone(a_, b_, c_)


class WithEvalCell(Cell):
    r"""
    Cell that returns loss, output and label for evaluation.

    This Cell accepts a network and loss function as arguments and computes loss for model.
    It returns loss, output and label to calculate the metrics.

    Args:
        network (Cell): The network Cell.
        loss_fn (Cell): The loss Cell.
        add_cast_fp32 (bool): Adjust the data type to float32.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tuple, containing a scalar loss Tensor, a network output Tensor of shape :math:`(N, \ldots)`
        and a label Tensor of shape :math:`(N, \ldots)`.

    Raises:
        TypeError: If `add_cast_fp32` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> eval_net = nn.WithEvalCell(net, loss_fn)
    """

    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = validator.check_value_type("add_cast_fp32", add_cast_fp32, [bool], self.cls_name)

    def construct(self, data, label):
        outputs = self._network(data)
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        loss = self._loss_fn(outputs, label)
        return loss, outputs, label


class ParameterUpdate(Cell):
    """
    Cell that updates parameters.

    With this Cell, one can manually update `param` with the input `Tensor`.

    Args:
        param (Parameter): The parameter to be updated manually.

    Raises:
        KeyError: If parameter with the specified name does not exist.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> network = nn.Dense(3, 4)
        >>> param = network.parameters_dict()['weight']
        >>> update = nn.ParameterUpdate(param)
        >>> update.phase = "update_param"
        >>> weight = Tensor(np.arange(12).reshape((4, 3)), mindspore.float32)
        >>> output = update(weight)
    """

    def __init__(self, param):
        super(ParameterUpdate, self).__init__(auto_prefix=False)
        if not isinstance(param, Parameter):
            raise TypeError("`param` must be `Parameter`, but got {}".format(param))
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
        self.map_ = C.Map()
        self.params = tuple(params)
        self.broadcast = P.Broadcast(0)

    def construct(self):
        datatypes = self.map_(F.partial(_get_datatype), self.params)
        params = self.map_(F.partial(_cast_datatype, mstype.float32), self.params)
        params = self.broadcast(params)
        new_params = self.map_(F.partial(_cast_datatype), datatypes, params)
        return new_params
