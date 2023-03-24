# Copyright 2022 Huawei Technologies Co., Ltd
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
import math
import time
import logging
from typing import Optional, Tuple
import numpy as np
import pytest

import mindspore

import mindspore.ops as ops
from mindspore.ops.primitive import constexpr
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from mindspore.common.api import jit
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, _calculate_correct_fan, One

import mindspore.nn as nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import Adam

from mindspore.train import Model, Callback
from mindspore import context, ParameterTuple, set_seed
from mindspore.communication.management import get_group_size
import mindspore.dataset.engine as de

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
IGNORE_ID = -1
VOCAB_SIZE = 4233
BATCH_SIZE = 32
LABLE_LEN = 30
MEL_BINS = 80
OPTIM_LR = 0.0005
WARMUP_STEPS = 10000
ACCUM_GRAD = 4
MAX_EPOCH = 5

cast = P.Cast()
add_grads = C.MultitypeFuncGraph("add_grads")


@add_grads.register("Tensor", "Tensor")
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, mstype.float32)


update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, cast(grad, mstype.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


reciprocal = P.Reciprocal()
_grad_scale = C.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensorInner(grad.indices, grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)), grad.dense_shape)


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
        new_grad = ops.clip_by_value(
            grad, F.cast(F.tuple_to_array((-clip_value,)),
                         dt), F.cast(F.tuple_to_array((clip_value,)), dt)
        )
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainAccumulationAllReduceEachWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    For distribution mode, allreduce will be implemented after each sub-step and the trailing time
    will be overided by backend optimization pass.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                  batch_size * accumulation_steps. Default: 1.
    """

    def __init__(self, network, optimizer, scale_update_cell, accumulation_steps=1, enable_global_norm=False):
        super(TrainAccumulationAllReduceEachWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.accumulation_steps = accumulation_steps
        self.enable_global_norm = enable_global_norm
        self.one = Tensor([1], mstype.int32)
        self.zero = Tensor([0], mstype.int32)
        self.local_step = Parameter(initializer(0, [1], mstype.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
        self.cast = P.Cast()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.loss_scale = self.scale_sense

    @C.add_flags(has_effect=True)
    def construct(self, *inputs):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*inputs)

        scaling_sens = self.loss_scale

        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(
            is_accu_step, self.local_step + self.one, self.one)
        loss_broadcast = self.reshape(loss, (1,))
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss_broadcast, loss_broadcast)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        # alloc status and clear should be right before gradoperation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(
            loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)

        accu_grads = self.hyper_map(add_grads, self.accu_grads, grads)
        scaling = scaling_sens * self.degree * self.accumulation_steps
        grads = self.hyper_map(F.partial(_grad_scale, scaling), accu_grads)
        grads = self.grad_reducer(grads)

        overflow = self.get_overflow_status(status, grads)
        overflow = self.logical_or(self.not_equal(
            self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(
            is_accu_step, accu_overflow, self.zero)
        overflow = self.reshape(overflow, (()))

        if is_accu_step:
            accu_succ = self.update_accu_grads_(accu_grads)
        else:
            overflow = self.loss_scaling_manager(self.loss_scale, overflow)
            if not overflow:
                if self.enable_global_norm:
                    grads = C.clip_by_global_norm(grads, 1.0, None)
                else:
                    grads = self.clip_grads(grads)

                self.optimizer(grads)

            accu_succ = self.reset_accu_grads_()

        ret = (mean_loss, overflow, scaling_sens.value(), overflow)
        return F.depend(ret, accu_succ)

    @jit
    def update_accu_grads_(self, accu_grads):
        return self.hyper_map(update_accu_grads, self.accu_grads, accu_grads)

    @jit
    def clip_grads(self, grads):
        return self.hyper_map(
                        F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

    @jit
    def reset_accu_grads_(self):
        return self.hyper_map(reset_accu_grads, self.accu_grads)


class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, steps_size):
        super(TimeMonitor, self).__init__()
        self.step = 0
        self.steps_size = steps_size
        self.step_time = 0.0
        self.loss = 0.0

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_seconds = (time.time() - self.step_time) * 1000
        cb_params = run_context.original_args()
        # TrainOneStepWithLossScaleCell returns tuple while TrainOneStepCell returns loss directly
        self.loss = cb_params.net_outputs[0].asnumpy()
        overflow = cb_params.net_outputs[3]
        scale = cb_params.net_outputs[2]
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError("data_size must be positive int.")

        step_seconds = step_seconds / 1000

        if overflow:
            logging.warning(
                "Epoch: %d, Step: %d, Step Time: %s sec, Total Loss: %i, Overflow: %i, Scale: %i.",
                int(self.step / self.steps_size),
                self.step % self.steps_size,
                str(step_seconds)[:5],
                self.loss,
                overflow,
                scale,
            )
        else:
            logging.warning(
                "Epoch: %d, Step: %d, Step Time: %s sec, Total Loss: %i, Scale: %i.",
                int(self.step / self.steps_size),
                self.step % self.steps_size,
                str(step_seconds)[:5],
                self.loss,
                scale,
            )
        self.step += 1


def get_parameter_numel(net):
    num = np.array([np.prod(item.shape) for item in net.get_parameters()]).sum() / 1024 / 1024
    return str(num)[:5] + "M"


class KLDivLoss(nn.Cell):
    """Construct an KLDivLoss module."""

    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.log = ops.Log()
        self.mul = ops.Mul()

    def construct(self, p, q):
        log_term = self.log(q) - p
        kl_div = self.mul(q, log_term)
        return kl_div


class LabelSmoothingLoss(nn.Cell):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
        compute_type: default mindspore.float32
    """

    def __init__(self, size, padding_idx, smoothing, normalize_length=False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = KLDivLoss()
        self.padding_idx = padding_idx
        self.on_value = Tensor([1.0 - smoothing], dtype=mstype.float32)
        self.off_value = Tensor([smoothing / (size - 1)], dtype=mstype.float32)
        self.size = size  # vocab size
        self.normalize_length = normalize_length
        self.log_softmax = nn.LogSoftmax(1)
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.mul = ops.Mul()
        self.onehot = ops.OneHot(axis=-1)

    def construct(self, x, target, target_masks):
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (mindspore.Tensor): prediction (batch, seqlen, class)
            target (mindspore.Tensor):
                target sequence masked with self.padding_id (batch, seqlen)
            target_masks (mindspore.Tensor): target sequence masks to indicate
                the padding part

        Returns:
            mindspore.Tensor: The KL loss, scalar float value
        """
        batch_size = x.shape[0]
        x = x.view(-1, self.size)
        target = target.view(-1)
        target_masks = target_masks.view(-1)
        target_zeropad = self.cast(
            self.mul(target, target_masks), mstype.int32)  # avoid -1 index
        total = target_masks.sum()
        denom = total if self.normalize_length else batch_size
        true_dist = self.onehot(target_zeropad, self.size,
                                self.on_value, self.off_value)

        kl = self.criterion(self.log_softmax(x), true_dist)
        # mask the loss of padded part
        kl = self.mul(kl, self.expand_dims(target_masks, 1))

        return kl.sum() / denom


class ASRWarmupLR(LearningRateSchedule):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
            self,
            learninig_rate=0.001,
            warmup_steps=25000,
            start_steps=0,
    ):
        super(ASRWarmupLR, self).__init__()
        self.learninig_rate = learninig_rate
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.min = ops.Minimum()
        self.start_steps = start_steps

    def construct(self, global_step):
        step_num = global_step + self.start_steps
        warmup_percent = self.warmup_steps**0.5 * \
            self.min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
        current_lr = self.learninig_rate * warmup_percent

        return current_lr


class Conv2d(nn.Cell):
    r"""
    A self-defined layer norm operation using reduce sum and reduce mean

    Args:
        normalized_shape (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
        Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, has_bias=False, pad_mode="valid",
                 negative_slope=math.sqrt(5), mode="fan_in", nonlinerity="leaky_relu"):
        super(Conv2d, self).__init__()
        kaiming_uniform_0 = initializer(
            0.5,
            (out_channel, in_channel, kernel_size, kernel_size),
        )
        fan_in = _calculate_correct_fan(
            (out_channel, in_channel, kernel_size, kernel_size), mode=mode)
        scale = 1 / math.sqrt(fan_in)
        bias_init_0 = initializer(
            scale, [out_channel], mindspore.float32)
        self.conv2d = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            has_bias=has_bias,
            pad_mode=pad_mode,
            weight_init=kaiming_uniform_0,
            bias_init=bias_init_0,
        )

    def construct(self, x):
        out = self.conv2d(x)
        return out


@constexpr
def check_dense_input_shape(x, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if len(x) < 2:
        raise ValueError(
            f"{msg_prefix} dimension of 'x' should not be less than 2, but got {len(x)}.")


class CustomDense(nn.Dense):
    def __init__(self, in_channels, out_channels, weight_init="normal", bias_init="zeros", has_bias=True,
                 activation=None):
        """Initialize CustomDense."""
        super(CustomDense, self).__init__(in_channels, out_channels,
                                          weight_init, bias_init, has_bias, activation)
        self.dyn_shape = ops.TensorShape()
        self.cast = ops.Cast()

    def construct(self, x):
        x_shape = self.shape_op(x)
        if F.is_sequence_value_unknown(x_shape):
            x_dyn_shape = self.dyn_shape(x)
            x_dyn_shape = self.cast(x_dyn_shape, mstype.float32)
            if len(x_dyn_shape) != 2:
                new_shape = x_dyn_shape.copy()[1:]
                new_shape[0] = x_dyn_shape[0:1] * x_dyn_shape[1:2]
                new_shape = self.cast(new_shape, mstype.int64)
                x = self.reshape(x, new_shape)
            x = self.matmul(x, self.weight)
            if self.has_bias:
                x = self.bias_add(x, self.bias)
            if self.activation_flag:
                x = self.activation(x)
            if len(x_dyn_shape) != 2:
                out_shape = self.dyn_shape(x)
                out_shape = self.cast(out_shape, mstype.float32)
                x_dyn_shape[2] = out_shape[1:2]
                x_dyn_shape = self.cast(x_dyn_shape, mstype.int64)
                x = self.reshape(x, x_dyn_shape)
        else:
            check_dense_input_shape(x_shape, self.cls_name)
            if len(x_shape) != 2:
                x = self.reshape(x, (-1, x_shape[-1]))
            x = self.matmul(x, self.weight)
            if self.has_bias:
                x = self.bias_add(x, self.bias)
            if self.activation_flag:
                x = self.activation(x)
            if len(x_shape) != 2:
                out_shape = x_shape[:-1] + (-1,)
                x = self.reshape(x, out_shape)
        return x


class Dense(nn.Cell):
    r"""
    A self-defined layer norm operation using reduce sum and reduce mean

    Args:
        normalized_shape (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
        Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, in_channel, out_channel, has_bias=True, activation=None, negative_slope=math.sqrt(5),
                 mode="fan_in", nonlinearity="leaky_relu"):
        super(Dense, self).__init__()
        kaiming_uniform_0 = initializer(
            0.5, (out_channel, in_channel)
        )
        bias_init_0 = "zeros"
        if has_bias:
            fan_in = _calculate_correct_fan(
                (out_channel, in_channel), mode=mode)
            scale = 1 / math.sqrt(fan_in)
            bias_init_0 = initializer(
                scale, [out_channel], mindspore.float32)
        self.dense = CustomDense(
            in_channel,
            out_channel,
            weight_init=kaiming_uniform_0,
            bias_init=bias_init_0,
            has_bias=True,
            activation=activation,
        )

    def construct(self, x):
        out = self.dense(x)
        return out


class CustomLayerNorm(nn.Cell):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm((normalized_shape,), epsilon=epsilon)
        self.cast = ops.Cast()
        self.get_dtype = ops.DType()

    def construct(self, x):
        output = self.cast(x, mstype.float32)
        output = self.layernorm(output)
        output = self.cast(output, self.get_dtype(x))

        return output


class TransformerEncoderLayer(nn.Cell):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (minspore.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (minspore.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)

    """

    def __init__(
            self,
            size,
            self_attn,
            feed_forward,
            dropout_rate=0.1,
            normalize_before=True,
            concat_after=False,
            compute_type=mstype.float32,
    ):
        """Construct an EncoderLayer object."""
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = CustomLayerNorm(size, epsilon=1e-5)
        self.norm2 = CustomLayerNorm(size, epsilon=1e-5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Dense(
                size + size, size).to_float(compute_type)
        self.cat_f1 = ops.Concat(-1)
        self.cat_1 = ops.Concat(1)

    def construct(
            self,
            x,
            mask,
            pos_emb,
            output_cache: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Compute encoded features.

        Args:
            x (minspore.Tensor): Input tensor (#batch, time, size).
            mask (minspore.Tensor): Mask tensor for the input (#batch, 1, time).
            pos_emb (minspore.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            output_cache (minspore.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
        Returns:
            minspore.Tensor: Output tensor (#batch, time, size).
            minspore.Tensor: Mask tensor (#batch, time).

        """
        # Multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        if self.concat_after:
            x_concat = self.cat_f1((x, self.self_attn(x_q, x, x, mask)))
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        # Feedforawrd module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = self.cat_1([output_cache, x], dim=1)

        return x, mask


class BaseEncoder(nn.Cell):
    def __init__(
            self,
            input_size,
            output_size=256,
            positional_dropout_rate=0.1,
            input_layer="conv2d",
            pos_enc_layer_type="abs_pos",
            normalize_before=True,
            feature_norm=True,
            compute_type=mindspore.float32,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        """
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        self.input_layer = input_layer
        if input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
            self.embed = subsampling_class(
                input_size, output_size, pos_enc_class(
                    output_size, positional_dropout_rate), compute_type
            )
        else:
            self.embed = pos_enc_class(output_size, positional_dropout_rate)

        self.normalize_before = normalize_before
        if normalize_before:
            self.after_norm = CustomLayerNorm(output_size, epsilon=1e-5)

        self.feature_norm = feature_norm
        if feature_norm:
            self.feature_norm = CustomLayerNorm(input_size, epsilon=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def construct(
            self,
            xs,
            masks,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            masks: masks for the input xs ()
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        if self.feature_norm:
            xs = self.feature_norm(xs)

        # masks is subsampled to (B, 1, T/subsample_rate)
        if self.input_layer == "conv2d":
            xs, pos_emb, masks = self.embed(xs, masks)
        else:
            xs, pos_emb = self.embed(xs)
        for layer in self.encoders:
            xs, masks = layer(xs, masks, pos_emb)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(
            self,
            input_size,
            output_size=256,
            attention_heads=4,
            linear_units=2048,
            num_blocks=12,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.0,
            input_layer="conv2d",
            pos_enc_layer_type="abs_pos",
            normalize_before=True,
            feature_norm=True,
            concat_after=False,
            activation_type="relu",
            compute_type=mstype.float32,
    ):
        """Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(
            input_size,
            output_size,
            positional_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            feature_norm,
            compute_type,
        )
        activation = nn.ReLU()

        self.encoders = nn.CellList(
            [
                TransformerEncoderLayer(
                    output_size,
                    MultiHeadedAttention(
                        attention_heads,
                        output_size,
                        attention_dropout_rate,
                        compute_type,
                    ),
                    PositionwiseFeedForward(
                        output_size,
                        linear_units,
                        dropout_rate,
                        activation,
                        compute_type,
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    compute_type,
                )
                for _ in range(num_blocks)
            ]
        )


class DecoderLayer(nn.Cell):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
            self,
            size,
            self_attn,
            src_attn,
            feed_forward,
            dropout_rate,
            normalize_before=True,
            concat_after=False,
            compute_type=mstype.float32,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = CustomLayerNorm(size, epsilon=1e-12)
        self.norm2 = CustomLayerNorm(size, epsilon=1e-12)
        self.norm3 = CustomLayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = Dense(
                size + size, size).to_float(compute_type)
            self.concat_linear2 = Dense(
                size + size, size).to_float(compute_type)
        self.cat1 = ops.Concat(axis=-1)
        self.cat2 = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()

    def construct(
            self,
            tgt,
            tgt_mask,
            memory,
            memory_mask,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Compute decoded features.

        Args:
            tgt (mindspore.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (mindspore.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (mindspore.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (mindspore.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (mindspore.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            mindspore.Tensor: Output tensor (#batch, maxlen_out, size).
            mindspore.Tensor: Mask for output tensor (#batch, maxlen_out).
            mindspore.Tensor: Encoded memory (#batch, maxlen_in, size).
            mindspore.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # Self-attention module
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        tgt_q = tgt
        tgt_q_mask = tgt_mask
        if self.concat_after:
            tgt_concat = self.cat1(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)))
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + \
                self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # Src-attention module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        if self.concat_after:
            x_concat = self.cat1(
                (x, self.src_attn(x, memory, memory, memory_mask)))
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + \
                self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        # Feedforward module
        residual = x
        if self.normalize_before:
            x = self.norm3(x)

        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        output_list = (x, tgt_mask, memory, memory_mask)
        return output_list


class TransformerDecoder(nn.Cell):
    """Base class of Transformer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
            self,
            vocab_size,
            encoder_output_size,
            attention_heads=4,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            self_attention_dropout_rate=0.0,
            src_attention_dropout_rate=0.0,
            input_layer="embed",
            use_output_layer=True,
            normalize_before=True,
            concat_after=False,
            compute_type=mstype.float32,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        activation = nn.ReLU()
        self.first_flag = True

        if input_layer == "embed":
            self.embed = nn.SequentialCell(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.use_output_layer = use_output_layer

        if use_output_layer:
            self.output_layer = Dense(
                attention_dim, vocab_size).to_float(compute_type)
        if normalize_before:
            self.after_norm = CustomLayerNorm(attention_dim, epsilon=1e-12)

        self.decoders = nn.CellList(
            [
                DecoderLayer(
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads,
                        attention_dim,
                        self_attention_dropout_rate,
                        compute_type,
                    ),
                    MultiHeadedAttention(
                        attention_heads,
                        attention_dim,
                        src_attention_dropout_rate,
                        compute_type,
                    ),
                    PositionwiseFeedForward(
                        attention_dim,
                        linear_units,
                        dropout_rate,
                        activation,
                        compute_type,
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    compute_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.expand_dims = ops.ExpandDims()
        self.log_softmax = nn.LogSoftmax()

    def construct(
            self,
            memory,
            memory_mask,
            ys_in_pad,
            ys_masks,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            masks:
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                mindspore.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        x, _ = self.embed(ys_in_pad)
        for layer in self.decoders:
            x, ys_masks, memory, memory_mask = layer(
                x, ys_masks, memory, memory_mask)

        if self.normalize_before:
            x = self.after_norm(x)

        if self.use_output_layer:
            x = self.output_layer(x)

        return x


class PositionwiseFeedForward(nn.Cell):
    """Positionwise feed forward layer.

    FeedForward are applied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimension.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation function.
    """

    def __init__(
            self,
            idim,
            hidden_units,
            dropout_rate,
            activation,
            compute_type=mstype.float32,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Dense(idim, hidden_units).to_float(compute_type)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)
        self.w_2 = Dense(hidden_units, idim).to_float(compute_type)

    def construct(self, xs):
        """Forward function.

        Args:
            xs (mindspore.Tensor): Input tensor (B, L, D)
        Returns:
            mindspore.Tensor: Output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class BaseSubsampling(nn.Cell):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset, size):
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, odim, pos_enc_class, compute_type=mindspore.float32):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = nn.SequentialCell(
            Conv2d(1, odim, 3, 2, has_bias=True, pad_mode="valid"),
            nn.ReLU(),
            Conv2d(odim, odim, 3, 2, has_bias=True, pad_mode="valid"),
            nn.ReLU(),
        )  # .to_float(compute_type) # dynamic-shaped Conv2D does not support float16 inputs currently
        self.compute_type = compute_type
        self.out = Dense(odim * (((idim - 1) // 2 - 1) // 2),
                         odim).to_float(compute_type)
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by: (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6
        self.expanddims = ops.ExpandDims()
        self.cast = ops.Cast()
        self.get_shape = ops.Shape()

    def construct(
            self, x, masks, offset=0
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Subsample x.

        Args:
            x (minspore.Tensor): Input tensor (#batch, time, idim).
            x_mask (minspore.Tensor): Input mask (#batch, 1, time).

        Returns:
            minspore.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            minspore.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            minspore.Tensor: positional encoding

        """
        x = self.expanddims(x, 1)  # (b, c=1, t, f)
        x = self.conv(x)
        x_shape = self.get_shape(x)
        b, c, _, f = x_shape
        x = self.out(x.transpose(0, 2, 1, 3).view(b, -1, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, masks


class PositionalEncoding(nn.Cell):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(
            self,
            d_model,
            dropout_rate,
            max_len=5000,
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = Tensor([math.sqrt(self.d_model)], dtype=mstype.float32)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = np.zeros((self.max_len, self.d_model))
        position = np.expand_dims(
            np.arange(0, self.max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(
            0, self.d_model, 2, dtype=np.float32) * -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Tensor(np.expand_dims(self.pe, 0), mstype.float32)
        self.get_shape = ops.Shape()
        self.dyn_shape = ops.TensorShape()

    def construct(self, x, offset=0) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        x_shape = self.get_shape(x)
        if F.is_sequence_value_unknown(x_shape):
            x_shape = self.dyn_shape(x)
        pos_emb = self.pe[:, offset: offset + x_shape[1]]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset, size):
        return self.dropout(self.pe[:, offset: offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in <https://arxiv.org/abs/1901.02860>
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def construct(self, x, offset=0) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Compute positional encoding.
        Args:
            x (minspore.Tensor): Input tensor (batch, time, `*`).
        Returns:
            minspore.Tensor: Encoded tensor (batch, time, `*`).
            minspore.Tensor: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        x_shape = self.get_shape(x)
        if F.is_sequence_value_unknown(x_shape):
            x_shape = self.dyn_shape(x)
        pos_emb = self.pe[:, offset: offset + x_shape[1]]
        return self.dropout(x), self.dropout(pos_emb)


class MultiHeadedAttention(nn.Cell):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
            self,
            n_head,
            n_feat,
            dropout_rate,
            compute_type=mstype.float32,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.neg_inf = Tensor([-10000.0], dtype=compute_type)
        self.scores_mul = Tensor(
            [1.0 / math.sqrt(float(self.d_k))], dtype=compute_type)

        self.linear_q = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_k = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_v = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_out = Dense(n_feat, n_feat).to_float(compute_type)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax()

        self.expand_dims = ops.ExpandDims()
        self.equal = ops.Equal()
        self.matmul = ops.BatchMatMul()
        self.cast = ops.Cast()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.get_dtype = ops.DType()

    def forward_qkv(
            self, query, key, value
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Transform query, key and value.

        Args:
            query (mindspore.Tensor): Query tensor (#batch, time1, size).
            key (mindspore.Tensor): Key tensor (#batch, time2, size).
            value (mindspore.Tensor): Value tensor (#batch, time2, size).

        Returns:
            mindspore.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            mindspore.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            mindspore.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.shape[0]
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(0, 2, 1, 3)  # (batch, head, time1, d_k)
        k = k.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)
        v = v.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
            self,
            value,
            scores,
            mask: Optional[mindspore.Tensor],
    ):
        """Compute attention context vector.

        Args:
            value (mindspore.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (mindspore.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (mindspore.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            mindspore.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.shape[0]

        if mask is not None:
            attn_mask = self.expand_dims(mask, 1)
            attn_mask = self.cast(self.equal(
                attn_mask, 0), self.get_dtype(scores))
            if len(attn_mask.shape) == 3:
                attn_mask = self.expand_dims(attn_mask, 1)
            attn_mask = self.mul(attn_mask, self.neg_inf)
            scores = self.add(attn_mask, scores)
            attn = self.softmax(scores)  # (batch, head, time1, time2)
        else:
            attn = self.softmax(scores)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        x = self.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(0, 2, 1, 3).view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)  # (batch, time1, d_model)

    def construct(
            self,
            query,
            key,
            value,
            mask: Optional[mindspore.Tensor],
    ):
        """Compute scaled dot product attention.

        Args:
            query (mindspore.Tensor): Query tensor (#batch, time1, size).
            key (mindspore.Tensor): Key tensor (#batch, time2, size).
            value (mindspore.Tensor): Value tensor (#batch, time2, size).
            mask (mindspore.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            mindspore.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = self.matmul(q * self.scores_mul,
                             k.transpose(0, 1, 3, 2) * self.scores_mul)

        return self.forward_attention(v, scores, mask)


class ASRModelWithAcc(nn.Cell):
    """CTC-attention hybrid encoder-decoder model"""

    def __init__(
            self,
            vocab_size,
            encoder=TransformerEncoder,
            decoder=TransformerDecoder,
            ctc=None,
            ctc_weight=0.5,
            ignore_id=IGNORE_ID,
            reverse_weight=0.0,
            lsm_weight=0.0,
            length_normalized_loss=False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.cast = ops.Cast()
        self.div = ops.Div()
        self.equal = ops.Equal()
        self.mul = ops.Mul()
        self.print = ops.Print()
        self.expand_dims = ops.ExpandDims()
        self.tile = ops.Tile()
        self.topk = ops.TopK()
        self.gather = ops.Gather()
        self.cat = ops.Concat(axis=1)

    def construct(
            self,
            xs_pad,
            xs_masks,
            ys_pad,
            ys_in_pad,
            ys_out_pad,
            ys_lengths,
            ys_masks,
            ys_sub_masks,
    ):
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(xs_pad, xs_masks)
        encoder_out = self.cast(encoder_out, mstype.float32)
        encoder_mask = self.cast(encoder_mask, mstype.float32)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(
                encoder_out,
                encoder_mask,
                ys_in_pad,
                ys_out_pad,
                ys_masks,
                ys_sub_masks,
            )
        else:
            loss_att = None
            acc_att = None

        # 2b. CTC branch
        loss_ctc = None

        # 3. final loss
        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + \
                (1 - self.ctc_weight) * loss_att
        return loss, acc_att

    def _calc_att_loss(
            self,
            encoder_out,
            encoder_mask,
            ys_in_pad,
            ys_out_pad,
            ys_masks,
            ys_sub_masks,
    ):
        # 1. Forward decoder
        decoder_out = self.decoder(encoder_out, encoder_mask, ys_in_pad, ys_sub_masks)
        decoder_out = self.cast(decoder_out, mstype.float32)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad, ys_masks)

        # 3. Compute attention accuracy
        acc_att = self._th_accuracy(
            decoder_out,
            ys_out_pad,
            ys_masks,
        )

        return loss_att, acc_att

    def _th_accuracy(self, pad_outputs, pad_targets, ys_masks):
        """Calculate accuracy.

        Args:
            pad_outputs (mindspore.Tensor): Prediction tensors (B * Lmax, D).
            pad_targets (mindspore.Tensor): Target label tensors (B, Lmax).
            ys_masks (mindspord.Tensor): Target label mask (B, Lmax)
            ignore_label (int): Ignore label id.

        Returns:
            mindspore.Tensor: Accuracy value (0.0 - 1.0).

        """
        pad_pred = pad_outputs.argmax(2)
        ys_masks = ys_masks.squeeze(1)
        numerator = self.mul(self.equal(pad_pred, pad_targets), ys_masks).sum()
        denominator = ys_masks.sum()
        return self.div(numerator, denominator)


class ASRModel(nn.Cell):
    """CTC-attention hybrid encoder-decoder model"""

    def __init__(self, vocab_size, encoder=TransformerEncoder, decoder=TransformerDecoder, ctc=None,
                 ctc_weight=0.0, ignore_id=IGNORE_ID, reverse_weight=0.0,
                 lsm_weight=0.1, length_normalized_loss=False):
        super().__init__()
        self.acc_net = ASRModelWithAcc(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_weight=ctc_weight,
            ignore_id=ignore_id,
            reverse_weight=reverse_weight,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
        )

    def construct(self, xs_pad, xs_masks, ys_pad,
                  ys_in_pad, ys_out_pad, ys_lengths,
                  ys_masks, ys_sub_masks):
        loss, _ = self.acc_net(xs_pad, xs_masks, ys_pad, ys_in_pad,
                               ys_out_pad, ys_lengths, ys_masks, ys_sub_masks)
        return loss


def init_asr_model(input_dim, vocab_size):
    compute_type = mstype.float16
    encoder = TransformerEncoder(
        input_dim,
        num_blocks=1,
        compute_type=compute_type,
    )

    decoder = TransformerDecoder(
        vocab_size,
        encoder.output_size(),
        num_blocks=1,
        compute_type=compute_type,
    )

    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=None,
    )

    return model


def create_dataset(batch_size=32, label_len=30, mel_bins=80):
    seq_len_list = [701, 474, 314, 502, 288, 629]
    data_list = []
    upper_limit = 50
    lower_limit = 25
    for seq_len in seq_len_list:
        mask_len = int(seq_len // 4) - 1
        xs_pad = np.random.randn(batch_size, seq_len, mel_bins).astype(np.float32)
        xs_masks = np.random.randint(0, 2, (batch_size, 1, mask_len)).astype(np.float32)
        ys_pad = np.random.randint(0, 10, (batch_size, label_len)).astype(np.int32)
        ys_in_pad = np.random.randint(0, 10, (batch_size, label_len + 1)).astype(np.int32)
        ys_out_pad = np.random.randint(0, 10, (batch_size, label_len + 1)).astype(np.int32)
        ys_lengths = \
            np.random.randint(int(seq_len // upper_limit), int(seq_len // lower_limit), (batch_size,)).astype(np.int32)
        ys_masks = np.random.randint(0, 2, (batch_size, 1, label_len + 1)).astype(np.float32)
        ys_sub_masks = np.random.randint(0, 2, (batch_size, label_len + 1, label_len + 1)).astype(np.float32)
        data_list.append((xs_pad, xs_masks, ys_pad, ys_in_pad, ys_out_pad, ys_lengths, ys_masks, ys_sub_masks))

    ds = de.GeneratorDataset(
        data_list,
        ["xs_pad", "xs_masks", "ys_pad", "ys_in_pad", "ys_out_pad", "ys_lengths", "ys_masks", "ys_sub_masks"],
    )
    return ds


def get_train_loss(train_dataset, run_mode):
    context.set_context(mode=run_mode, device_target="Ascend")
    bs = BATCH_SIZE
    ll = LABLE_LEN
    mb = MEL_BINS
    steps_size = train_dataset.get_dataset_size()
    logging.warning("Training dataset has %d steps in each epoch.", steps_size)

    # define wenet network
    wenet_with_loss = init_asr_model(mb, VOCAB_SIZE)
    weights = ParameterTuple(wenet_with_loss.trainable_params())
    logging.info("Total parameter of WeNet-ASR model: %s.",
                 get_parameter_numel(wenet_with_loss))

    lr_schedule = ASRWarmupLR(
        learninig_rate=OPTIM_LR,
        warmup_steps=WARMUP_STEPS,
    )
    optimizer = Adam(weights, learning_rate=lr_schedule)

    update_cell = DynamicLossScaleUpdateCell(
        loss_scale_value=1024, scale_factor=2, scale_window=1000)

    train_net = TrainAccumulationAllReduceEachWithLossScaleCell(
        wenet_with_loss, optimizer, update_cell, accumulation_steps=ACCUM_GRAD
    )

    callback = TimeMonitor(steps_size)
    xs_pad = Tensor(shape=[bs, None, mb], dtype=mindspore.float32)
    xs_masks = Tensor(shape=[bs, 1, None], dtype=mindspore.float32)
    ys_pad = Tensor(shape=[bs, ll], dtype=mindspore.int32, init=One())
    ys_in_pad = Tensor(shape=[bs, ll + 1], dtype=mindspore.int32, init=One())
    ys_out_pad = Tensor(shape=[bs, ll + 1], dtype=mindspore.int32, init=One())
    ys_lengths = Tensor(shape=[bs,], dtype=mindspore.int32, init=One())
    ys_masks = Tensor(shape=[bs, 1, ll + 1],
                      dtype=mindspore.float32, init=One())
    ys_sub_masks = Tensor(shape=[bs, ll + 1, ll + 1],
                          dtype=mindspore.float32, init=One())
    train_net.set_inputs(xs_pad, xs_masks, ys_pad, ys_in_pad,
                         ys_out_pad, ys_lengths, ys_masks, ys_sub_masks)

    model = Model(train_net)
    logging.info("Training start.")

    model.train(
        MAX_EPOCH,
        train_dataset,
        callbacks=callback,
        dataset_sink_mode=True
    )
    return callback.loss


def train_proccess(mode):
    logging.info("Initializing training dataset.")
    bs = BATCH_SIZE
    ll = LABLE_LEN
    mb = MEL_BINS
    set_seed(0)
    train_dataset = create_dataset(bs, ll, mb)
    expect_graph_loss = get_train_loss(train_dataset, mode)
    assert np.allclose(expect_graph_loss, 111.1163, 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_train():
    """
    Feature: Test the simplified dynamic shape WeNet-ASR network with small data.
    Description:  The sequence length of inputs is dynamic.
    Expectation: Assert that the training loss of fixed data is consistent with the expected loss.
    """
    train_proccess(context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_train_pynative():
    """
    Feature: Test the simplified dynamic shape WeNet-ASR network with small data.
    Description:  The sequence length of inputs is dynamic.
    Expectation: Assert that the training loss of fixed data is consistent with the expected loss.
    """
    train_proccess(context.PYNATIVE_MODE)
