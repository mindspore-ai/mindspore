# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Transformer for training."""
from mindspore import jit

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size

from tests.st.dynamic_shape.transformer.transformer_model import TransformerModel

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0
VOCAB_SIZE = 36560
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 32

clip_grad = ops.MultitypeFuncGraph("clip_grad")


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
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt),
                                     ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


class TransformerTrainingLoss(nn.Cell):
    """
    Provide transformer training loss.

    Args:
        is_graph_mode (bool): is graph mode.

    Returns:
        Tensor, total loss.
    """
    def __init__(self, is_graph_mode):
        super(TransformerTrainingLoss, self).__init__(auto_prefix=False)
        self.vocab_size = VOCAB_SIZE
        self.onehot = ops.OneHot()
        self.on_value = Tensor(float(1 - LABEL_SMOOTHING), ms.float32)
        self.off_value = Tensor(LABEL_SMOOTHING / float(self.vocab_size - 1), ms.float32)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.reshape = ops.Reshape()
        self.last_idx = (-1,)
        self.flatten = ops.Flatten()
        self.neg = ops.Neg()
        self.cast = ops.Cast()
        self.batch_size = BATCH_SIZE
        self.is_graph_mode = is_graph_mode

    def construct(self, prediction_scores, label_ids, label_weights, seq_length):
        """Defines the computation performed."""
        flat_shape = (self.batch_size * seq_length,)
        label_ids = self.reshape(label_ids, flat_shape)
        label_weights = self.cast(self.reshape(label_weights, flat_shape), ms.float32)
        one_hot_labels = self.onehot(self.cast(label_ids, ms.int32), self.cast(self.vocab_size, ms.int32),
                                     self.on_value, self.off_value)

        per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx))
        numerator = self.reduce_sum(label_weights * per_example_loss, ())
        denominator = self.reduce_sum(label_weights, ()) + \
                      self.cast(ops.tuple_to_array((1e-5,)), ms.float32)
        loss = numerator / denominator
        return loss


class TransformerNetworkWithLoss(nn.Cell):
    """
    Provide  transformer training loss through network.

    Args:
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.
        is_graph_mode (bool): is graph mode.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, is_training, use_one_hot_embeddings=False, is_graph_mode=False):
        super(TransformerNetworkWithLoss, self).__init__(auto_prefix=False)
        self.transformer = TransformerModel(is_training, use_one_hot_embeddings, is_graph_mode)
        self.loss = TransformerTrainingLoss(is_graph_mode)
        self.cast = ops.Cast()
        self.shape = ops.Shape()

    def construct(self,
                  source_ids,
                  source_mask,
                  target_ids,
                  target_mask,
                  label_ids,
                  label_weights):
        """Transformer network with loss."""
        prediction_scores = self.transformer(source_ids, source_mask, target_ids, target_mask)
        seq_length = self.shape(source_ids)[1]
        total_loss = self.loss(prediction_scores, label_ids, label_weights, seq_length)
        return self.cast(total_loss, ms.float32)


grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))

_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class TransformerTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of Transformer network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(TransformerTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = ops.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ms.float32))
        self.enable_tuple_broaden = True

    @jit
    def clip_grads(self, grads):
        grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads

    @jit
    def clip_scale_grads(self, scale, grads):
        grads = self.hyper_map(ops.partial(grad_scale, scale * self.degree), grads)
        return grads

    def construct(self,
                  source_eos_ids,
                  source_eos_mask,
                  target_sos_ids,
                  target_sos_mask,
                  target_eos_ids,
                  target_eos_mask,
                  sens=None):
        """Defines the computation performed."""
        source_ids = source_eos_ids
        source_mask = source_eos_mask
        target_ids = target_sos_ids
        target_mask = target_sos_mask
        label_ids = target_eos_ids
        label_weights = target_eos_mask

        weights = self.weights
        loss = self.network(source_ids,
                            source_mask,
                            target_ids,
                            target_mask,
                            label_ids,
                            label_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(source_ids,
                                                 source_mask,
                                                 target_ids,
                                                 target_mask,
                                                 label_ids,
                                                 label_weights,
                                                 self.cast(scaling_sens,
                                                           ms.float32))

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.clip_scale_grads(scaling_sens, grads)
        grads = self.clip_grads(grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond, scaling_sens.value())


cast = ops.Cast()
add_grads = ops.MultitypeFuncGraph("add_grads")


@add_grads.register("Tensor", "Tensor")
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, ms.float32)

update_accu_grads = ops.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return ops.depend(succ, ops.assign(accu_grad, cast(grad, ms.float32)))

accumulate_accu_grads = ops.MultitypeFuncGraph("accumulate_accu_grads")


@accumulate_accu_grads.register("Tensor", "Tensor")
def _accumulate_accu_grads(accu_grad, grad):
    succ = True
    return ops.depend(succ, ops.assign_add(accu_grad, cast(grad, ms.float32)))


zeroslike = ops.ZerosLike()
reset_accu_grads = ops.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return ops.depend(succ, ops.assign(accu_grad, zeroslike(accu_grad)))
