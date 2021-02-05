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
from mindspore import nn
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean

from .transformer import Transformer
from .grad_clip import GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE, ClipGradients


class PredLogProbs(nn.Cell):
    """
    Get log probs.

    Args:
        config (TransformerConfig): The config of Transformer.

    Returns:
        Tensor, masked lm output.
    """

    def __init__(self, config):
        super(PredLogProbs, self).__init__()
        self.width = config.hidden_size
        self.reshape = P.Reshape()

        self.matmul = P.MatMul(transpose_b=True)
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.shape_flat_sequence_tensor = (config.batch_size * config.seq_length, self.width)
        self.cast = P.Cast()
        self.compute_type = config.compute_type
        self.dtype = config.dtype
        self.get_shape = P.Shape()

    def construct(self, input_tensor, output_weights):
        """
        Construct network.

        Args:
            input_tensor (Tensor): Tensor.
            output_weights (Tensor): Tensor.

        Returns:
            Tensor, masked lm output.
        """
        shape = self.get_shape(input_tensor)

        input_tensor = self.reshape(input_tensor, (shape[0] * shape[1], shape[2]))
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)

        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)

        log_probs = self.log_softmax(logits)
        return log_probs


class TransformerTraining(nn.Cell):
    """
    Transformer training network.

    Args:
        config (TransformerConfig): The config of Transformer.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores, seq_relationship_score.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(TransformerTraining, self).__init__()
        self.transformer = Transformer(config, is_training, use_one_hot_embeddings)
        self.projection = PredLogProbs(config)

    def construct(self, source_ids, source_mask, target_ids, target_mask):
        """
        Construct network.

        Args:
            source_ids (Tensor): Source sentence.
            source_mask (Tensor): Source padding mask.
            target_ids (Tensor): Target sentence.
            target_mask (Tensor): Target padding mask.

        Returns:
            Tensor, prediction_scores, seq_relationship_score.
        """
        _, decoder_outputs, embedding_table = \
            self.transformer(source_ids, source_mask, target_ids, target_mask)
        prediction_scores = self.projection(decoder_outputs,
                                            embedding_table)
        return prediction_scores


class LabelSmoothedCrossEntropyCriterion(nn.Cell):
    """
    Label Smoothed Cross-Entropy Criterion.

    Args:
        config (TransformerConfig): The config of Transformer.

    Returns:
        Tensor, final loss.
    """

    def __init__(self, config):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
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
        self.flat_shape = (config.batch_size * config.seq_length,)
        self.get_shape = P.Shape()

    def construct(self, prediction_scores, label_ids, label_weights):
        """
        Construct network to calculate loss.

        Args:
            prediction_scores (Tensor): Prediction scores.
            label_ids (Tensor): Labels.
            label_weights (Tensor): Mask tensor.

        Returns:
            Tensor, final loss.
        """
        label_shape = self.get_shape(label_ids)

        label_ids = self.reshape(label_ids, (label_shape[0] * label_shape[1],))
        label_weights = self.cast(
            self.reshape(label_weights, (label_shape[0] * label_shape[1],)),
            mstype.float32
        )
        one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)

        per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx))
        numerator = self.reduce_sum(label_weights * per_example_loss, ())
        denominator = self.reduce_sum(label_weights, ()) + self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
        loss = numerator / denominator

        return loss


class TransformerNetworkWithLoss(nn.Cell):
    """
    Provide  transformer training loss through network.

    Args:
        config (BertConfig): The config of Transformer.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(TransformerNetworkWithLoss, self).__init__()
        self.transformer = TransformerTraining(config, is_training, use_one_hot_embeddings)
        self.loss = LabelSmoothedCrossEntropyCriterion(config)
        self.cast = P.Cast()

    def construct(self,
                  source_ids,
                  source_mask,
                  target_ids,
                  target_mask,
                  label_ids,
                  label_weights):
        prediction_scores = self.transformer(source_ids, source_mask, target_ids, target_mask)
        total_loss = self.loss(prediction_scores, label_ids, label_weights)
        return self.cast(total_loss, mstype.float32)


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

class TransformerTrainOneStepWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of Transformer network training.

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

        super(TransformerTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
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
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.gpu_target = False
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()

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
        """
        Construct network.

        Args:
            source_eos_ids (Tensor): Source sentence.
            source_eos_mask (Tensor): Source padding mask.
            target_sos_ids (Tensor): Target sentence.
            target_sos_mask (Tensor): Target padding mask.
            target_eos_ids (Tensor): Prediction sentence.
            target_eos_mask (Tensor): Prediction padding mask.
            sens (Tensor): Loss sen.

        Returns:
            Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.
        """
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

        init = False
        if not self.gpu_target:
            # init overflow buffer
            init = self.alloc_status()
            init = F.depend(init, loss)
            # clear overflow buffer
            clear_status = self.clear_status(init)
            scaling_sens = F.depend(scaling_sens, clear_status)

        grads = self.grad(self.network, weights)(source_ids,
                                                 source_mask,
                                                 target_ids,
                                                 target_mask,
                                                 label_ids,
                                                 label_weights,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))

        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)

        if self.reducer_flag:
            # Apply grad reducer on grads.
            grads = self.grad_reducer(grads)

        # get the overflow buffer
        if not self.gpu_target:
            init = F.depend(init, grads)
            get_status = self.get_status(init)
            init = F.depend(init, get_status)
            # sum overflow buffer elements, 0:not overflow , >0:overflow
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            # convert flag_sum to scalar
            flag_sum = self.reshape(flag_sum, (()))

        if self.is_distributed:
            # Sum overflow flag over devices.
            flag_reduce = self.all_reduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)

        ret = (loss, cond, scaling_sens)
        return F.depend(ret, succ)
