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
"""CPM train module"""
import numpy as np

import mindspore.nn as nn
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops.operations.comm_ops import _VirtualDataset
from mindspore.ops import functional as F
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell

from src.cpm_loss import Cross_entropy
from src.cpm import CPMModel
from src.util import ClipByGlobalNorm


class CPMWithLoss(nn.Cell):
    """
    Provide  CPM training loss through network.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        vocab_size (int): Size of the vocabulary list.
        hidden_size (int): Internal feature dimension.
        config: The config of CPM network.
        num_hidden_layers (int): Number of hidden layers.
        num_attention_heads (int): Number of attention heads.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, batch_size, seq_length, vocab_size, hidden_size,
                 config, num_hidden_layers, num_attention_heads):
        super(CPMWithLoss, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.squeeze = P.Squeeze()
        self.expanddims = P.ExpandDims().shard(((config.dp, 1),))
        self.expanddims1 = P.ExpandDims().shard(((config.dp,),))
        self.tile = P.Tile().shard(((config.dp, 1, 1),))
        self.reducesum = P.ReduceSum().shard(((config.dp, 1, 1),))
        self.reducesum2 = P.ReduceSum().shard(((config.dp, 1),))
        self.reducemean = P.ReduceMean().shard(((1, 1),))
        self.cast = P.Cast()
        self.readdiv = P.RealDiv().shard(((config.dp, 1), (config.dp, 1)))
        self.readdiv2 = P.RealDiv().shard(((1,), (1,)))
        self.mul = P.Mul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.mul2 = P.Mul().shard(((config.dp, 1), (config.dp, 1)))

        self.cpm_model = CPMModel(batch_size=self.batch_size,
                                  seq_length=self.seq_length,
                                  vocab_size=self.vocab_size,
                                  hidden_size=self.hidden_size,
                                  config=config,
                                  hidden_dropout=config.dropout,
                                  attention_dropout=config.dropout,
                                  num_hidden_layers=self.num_hidden_layers,
                                  num_attention_heads=self.num_attention_heads,
                                  is_training=True)

        self.loss_net = Cross_entropy(batch_size=self.batch_size,
                                      seq_length=self.seq_length,
                                      vocab_size=self.vocab_size,
                                      config=config,
                                      is_training=True)
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))

    def construct(self, input_ids, attention_mask=None, position_ids=None, loss_mask=None, labels=None):
        r"""
        CPM model with loss.
        """
        input_ids = self.slice(input_ids, (0, 0),
                               (self.batch_size, self.seq_length),
                               (1, 1))
        position_ids = self.slice(position_ids, (0, 0),
                                  (self.batch_size, self.seq_length),
                                  (1, 1))
        attention_mask_1 = self.slice_mask(attention_mask, (0, 0, 0),
                                           (self.batch_size, self.seq_length, self.seq_length),
                                           (1, 1, 1))
        logist = self.cpm_model(input_ids, position_ids, attention_mask_1)
        loss_mask_expand = self.expanddims(loss_mask, -1)
        # 8 725 -> 8, 725, 1
        loss_masks = self.tile(loss_mask_expand, (1, 1, self.vocab_size))
        # 8 725 30000
        loss_mask_sum = self.expanddims1(self.reducesum2(loss_mask, -1), -1)
        # [8, 725, 30000|8, 725, 30000
        logist_mask_mul = self.mul(logist, loss_masks)
        # 8, 725, 30000->8, 30000
        logist_mask_sum = self.reducesum(logist_mask_mul, 1)
        # 8, 30000| 8 1
        output = self.readdiv(logist_mask_sum, loss_mask_sum)
        # 8 725 | 8 725
        label_mul_mask = self.mul2(labels, loss_mask)
        # 8 725 -> 8
        label_mask = self.reducesum2(label_mul_mask, 1)
        # 8 725 -> 8
        loss_mask_for_label = self.reducesum2(loss_mask, -1)
        # 8 / 8
        label_final = self.readdiv2(label_mask, loss_mask_for_label)
        # batch 1 vocabe_size
        output = self.expanddims(output, 1)
        # batchsize 1
        label_final = self.expanddims1(label_final, 1)
        # batchsize 1
        losses = self.loss_net(output, self.cast(label_final, mstype.float32))
        loss = self.reducemean(losses, 0)
        return loss


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
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
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
            F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad,
                                   F.cast(F.tuple_to_array((clip_value,)),
                                          dt))
    return new_grad


class VirtualDatasetOneInputCell(nn.Cell):
    def __init__(self, backbone):
        super(VirtualDatasetOneInputCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, *data):
        data_ = self._virtual_dataset(*data)
        return self._backbone(*data_)


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class CPMTrainOneStepWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of CPM network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        enable_global_norm (Bool): Whether using global normalization.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_update_cell=None,
                 enable_global_norm=True):
        super(CPMTrainOneStepWithLossScaleCell,
              self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.default_lr = Tensor([0.0], dtype=mstype.float32)
        self.enable_global_norm = enable_global_norm
        self.cast = P.Cast()
        self.clip = ClipByGlobalNorm(self.weights)

    def construct(self,
                  input_ids,
                  attention_mask,
                  position_ids,
                  loss_mask,
                  labels,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            attention_mask,
                            position_ids,
                            loss_mask,
                            labels)

        scaling_sens = self.scale_sense
        # alloc status and clear should be right before grad operation.
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network,
                          weights)(input_ids,
                                   attention_mask,
                                   position_ids,
                                   loss_mask,
                                   labels,
                                   scaling_sens_filled)
        # apply grad reducer on grads.
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(grad_scale, scaling_sens), grads)

        if self.enable_global_norm:
            grads, _ = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)

        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        return F.depend(loss, succ), cond, scaling_sens


cast = P.Cast()
update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign_add(accu_grad, cast(grad, mstype.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


class CPMTrainAccuStepsWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of CPM network training with loss scale.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        enable_global_norm (Bool): Whether using global normalization.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_update_cell=None,
                 enable_global_norm=True):
        super(CPMTrainAccuStepsWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.accumulation = False
        self.accumulation_steps = context.get_auto_parallel_context("grad_accumulation_step")
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))

        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))

        self.cast = P.Cast()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.enable_global_norm = enable_global_norm
        self.clip = ClipByGlobalNorm(self.weights)

    def construct(self,
                  input_ids,
                  attention_mask,
                  position_ids,
                  loss_mask,
                  labels,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            attention_mask,
                            position_ids,
                            loss_mask,
                            labels)
        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))

        grads = self.grad(self.network,
                          weights)(input_ids,
                                   attention_mask,
                                   position_ids,
                                   loss_mask,
                                   labels, scaling_sens_filled)

        if self.accumulation and self.accumulation_steps > 1:
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, grads)
            loss = F.depend(loss, accu_succ)

        overflow = self.get_overflow_status(status, grads)
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)

        if self.accumulation:
            succ = False
            self.accu_overflow = accu_overflow
        else:
            my_zero = F.depend(self.zero, accu_overflow)
            initialize = P.Assign()(self.accu_overflow, my_zero)
            grads1 = F.depend(grads, initialize)
            # apply grad reducer on grads
            grads = self.grad_reducer(grads1)

            scaling = scaling_sens * self.accumulation_steps
            grads = self.hyper_map(F.partial(grad_scale, scaling), grads)
            if self.enable_global_norm:
                grads, _ = self.clip(grads)
            else:
                grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
            accu_overflow = self.allreduce(accu_overflow)

            overflow = self.less_equal(self.base, accu_overflow)
            accu_grads = F.depend(self.accu_grads, grads)

            accu_succ = self.hyper_map(reset_accu_grads, accu_grads)
            overflow = F.depend(overflow, accu_succ)

            overflow = self.reshape(overflow, (()))
            overflow = self.process_loss_scale(overflow)

            if overflow:
                succ = False
            else:
                succ = self.optimizer(grads)

        return F.depend(loss, succ), overflow, scaling_sens
