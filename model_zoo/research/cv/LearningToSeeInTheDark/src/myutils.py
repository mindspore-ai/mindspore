# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Train one step with loss scale"""
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.
    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, label):
        """ construct of loss cell  """
        logits = self._backbone(x)
        return self._loss_fn(logits, label)

    @property
    def backbone_network(self):
        """
        Get the backbone network.
        Returns:
            Cell, return backbone network.
        """
        return self._backbone


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5
clip_grad = C.MultitypeFuncGraph("clip_grad")


class ClipGradients(nn.Cell):
    """
    Clip gradients.
    Returns:
        List, a list of clipped_grad tuples.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self, grads, clip_type, clip_value):
        """
        Construct gradient clip network.
        Args:
            grads (list): List of gradient tuples.
            clip_type (Tensor): The way to clip, 'value' or 'norm'.
            clip_value (Tensor): Specifies how much to clip.
        Returns:
            List, a list of clipped_grad tuples.
        """
        if clip_type not in (0, 1):
            return grads
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)
        return new_grads


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.
    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.
    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")

reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """ grad scale """
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")

grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class GNMTTrainOneStepWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of GNMT network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        network: Cell. The training network. Note that loss function should have
            been added.
        optimizer: Optimizer. Optimizer for updating the weights.
    Returns:
        Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(GNMTTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.all_reduce = P.AllReduce()
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode not in ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.clip_gradients = ClipGradients()
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell

        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))
        self.add_flags(has_effect=True)
        self.loss_scalar = P.ScalarSummary()

    def construct(self, inputs, labels, sens=None):
        """
            network processing
            overflow testing
        """
        weights = self.weights
        loss = self.network(inputs, labels)

        # Alloc status.
        init = self.alloc_status()

        # Clear overflow buffer.
        self.clear_before_grad(init)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        grads = self.grad(self.network, weights)(inputs, labels, self.cast(scaling_sens, mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)

        if self.reducer_flag:
            # Apply grad reducer on grads.
            grads = self.grad_reducer(grads)

        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))

        if self.is_distributed:
            # Sum overflow flag over devices.
            flag_reduce = self.all_reduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        overflow = cond

        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        self.loss_scalar("loss", loss)
        return (loss, cond, scaling_sens)
