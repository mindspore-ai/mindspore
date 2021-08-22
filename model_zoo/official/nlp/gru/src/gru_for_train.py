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
"""GRU train cell"""
from mindspore import Tensor, Parameter, ParameterTuple, context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size
from model_utils.config import config
from src.loss import NLLLoss

class GRUWithLossCell(nn.Cell):
    """
    GRU network connect with loss function.

    Args:
        network: The training network.

    Returns:
        the output of loss function.
    """
    def __init__(self, network):
        super(GRUWithLossCell, self).__init__()
        self.network = network
        self.loss = NLLLoss()
        self.logits_shape = (-1, config.src_vocab_size)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.mean = P.ReduceMean()
        self.text_len = config.max_length
        self.split = P.Split(axis=0, output_num=config.max_length-1)
        self.squeeze = P.Squeeze()
        self.add = P.AddN()
        self.transpose = P.Transpose()
        self.shape = P.Shape()
    def construct(self, encoder_inputs, decoder_inputs, teacher_force):
        '''
        GRU loss cell

        Args:
            encoder_inputs(Tensor): encoder inputs
            decoder_inputs(Tensor): decoder inputs
            teacher_force(Tensor): teacher force flag

        Returns:
            loss(scalar): loss output
        '''
        logits = self.network(encoder_inputs, decoder_inputs, teacher_force)
        logits = self.cast(logits, mstype.float32)
        loss_total = ()
        decoder_targets = decoder_inputs
        decoder_output = logits
        for i in range(1, self.text_len):
            loss = self.loss(self.squeeze(decoder_output[i-1:i:1, ::, ::]), decoder_targets[:, i])
            loss_total += (loss,)
        loss = self.add(loss_total) / self.text_len
        return loss

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Args:
        grads (list): List of gradient tuples.
        clip_type (Tensor): The way to clip, 'value' or 'norm'.
        clip_value (Tensor): Specifies how much to clip.

    Returns:
        List, a list of clipped_grad tuples.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()
    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """Defines the gradients clip."""
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
            t = self.cast(t, dt)
            new_grads = new_grads + (t,)
        return new_grads

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class GRUTrainOneStepWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of GRU network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(GRUTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()

        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode not in ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.clip_gradients = ClipGradients()
        self.cast = P.Cast()
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.gpu_target = False
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

    @C.add_flags(has_effect=True)
    def construct(self,
                  encoder_inputs,
                  decoder_inputs,
                  teacher_force,
                  sens=None):
        """Defines the computation performed."""

        weights = self.weights
        loss = self.network(encoder_inputs,
                            decoder_inputs,
                            teacher_force)
        init = False
        if not self.gpu_target:
            # alloc status
            init = self.alloc_status()
            # clear overflow buffer
            self.clear_before_grad(init)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        grads = self.grad(self.network, weights)(encoder_inputs,
                                                 decoder_inputs,
                                                 teacher_force,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))

        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        if not self.gpu_target:
            self.get_status(init)
            # sum overflow buffer elements, 0: not overflow, >0: overflow
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            # convert flag_sum to scalar
            flag_sum = self.reshape(flag_sum, (()))

        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond, scaling_sens)

class GRUTrainOneStepCell(nn.TrainOneStepCell):
    """
    Encapsulation class of GRU network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        enable_clip_grad (boolean): If True, clip gradients in GRUTrainOneStepCell. Default: True.
    """

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(GRUTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.clip_gradients = ClipGradients()
        self.enable_clip_grad = enable_clip_grad

    def set_sens(self, value):
        self.sens = value

    def construct(self,
                  encoder_inputs,
                  decoder_inputs,
                  teacher_force,
                  sens=None):
        """Defines the computation performed."""

        weights = self.weights
        loss = self.network(encoder_inputs,
                            decoder_inputs,
                            teacher_force)

        grads = self.grad(self.network, weights)(encoder_inputs,
                                                 decoder_inputs,
                                                 teacher_force,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        if self.enable_clip_grad:
            grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)
