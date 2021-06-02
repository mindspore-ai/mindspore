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
"""Transformer for training."""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size

from .transformer_model import TransformerModel

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


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
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class TransformerTrainingLoss(nn.Cell):
    """
    Provide transformer training loss.

    Args:
        config (TransformerConfig): The config of Transformer.

    Returns:
        Tensor, total loss.
    """
    def __init__(self, config):
        super(TransformerTrainingLoss, self).__init__(auto_prefix=False)
        self.vocab_size = config.vocab_size
        self.onehot = P.OneHot()
        self.on_value = Tensor(float(1 - config.label_smoothing), mstype.float32)
        self.off_value = Tensor(config.label_smoothing / float(self.vocab_size - 1), mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.flatten = P.Flatten()
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.batch_size = config.batch_size

    def construct(self, prediction_scores, label_ids, label_weights, seq_length):
        """Defines the computation performed."""
        flat_shape = (self.batch_size * seq_length,)
        label_ids = self.reshape(label_ids, flat_shape)
        label_weights = self.cast(self.reshape(label_weights, flat_shape), mstype.float32)
        one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)

        per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx))
        numerator = self.reduce_sum(label_weights * per_example_loss, ())
        denominator = self.reduce_sum(label_weights, ()) + \
                      self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
        loss = numerator / denominator
        return loss


class TransformerNetworkWithLoss(nn.Cell):
    """
    Provide  transformer training loss through network.

    Args:
        config (TransformerConfig): The config of Transformer.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(TransformerNetworkWithLoss, self).__init__(auto_prefix=False)
        self.transformer = TransformerModel(config, is_training, use_one_hot_embeddings)
        self.loss = TransformerTrainingLoss(config)
        self.cast = P.Cast()
        self.shape = P.Shape()

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
        return self.cast(total_loss, mstype.float32)


class TransformerTrainOneStepCell(nn.TrainOneStepCell):
    """
    Encapsulation class of transformer network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TransformerTrainOneStepCell, self).__init__(network, optimizer, sens)

        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def set_sens(self, value):
        self.sens = value

    def construct(self,
                  source_eos_ids,
                  source_eos_mask,
                  target_sos_ids,
                  target_sos_mask,
                  target_eos_ids,
                  target_eos_mask,):
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
        grads = self.grad(self.network, weights)(source_ids,
                                                 source_mask,
                                                 target_ids,
                                                 target_mask,
                                                 label_ids,
                                                 label_weights,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


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
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

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
                                                           mstype.float32))

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return F.depend(ret, succ)
