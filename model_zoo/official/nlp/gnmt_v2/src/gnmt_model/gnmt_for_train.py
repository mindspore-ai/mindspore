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
"""GNMT for training."""
import numpy as np

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

from .gnmt import GNMT
from .grad_clip import GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE, ClipGradients


class PredLogProbs(nn.Cell):
    """
    Get log probs.

    Args:
        config (GNMTConfig): The config of GNMT.

    Returns:
        Tensor, log softmax output.
    """

    def __init__(self, config):
        super(PredLogProbs, self).__init__()
        self.reshape = P.Reshape()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.get_shape = P.Shape()

    def construct(self, input_tensor):
        """
        Construct network.

        Args:
            input_tensor (Tensor): Tensor.

        Returns:
            Tensor, log softmax output.
        """
        shape = self.get_shape(input_tensor)
        logits = self.reshape(input_tensor, (shape[0] * shape[1], shape[2]))
        log_probs = self.log_softmax(logits)
        return log_probs


class GNMTTraining(nn.Cell):
    """
    GNMT training network.

    Args:
        config (GNMTConfig): The config of GNMT.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(GNMTTraining, self).__init__()
        self.gnmt = GNMT(config, is_training, use_one_hot_embeddings)
        self.projection = PredLogProbs(config)

    def construct(self, source_ids, source_mask, target_ids):
        """
        Construct network.

        Args:
            source_ids (Tensor): Source sentence.
            source_mask (Tensor): Source padding mask.
            target_ids (Tensor): Target sentence.

        Returns:
            Tensor, prediction_scores.
        """
        decoder_outputs = self.gnmt(source_ids, source_mask, target_ids)
        prediction_scores = self.projection(decoder_outputs)
        return prediction_scores


class LabelSmoothedCrossEntropyCriterion(nn.Cell):
    """
    Label Smoothed Cross-Entropy Criterion.

    Args:
        config (GNMTConfig): The config of GNMT.

    Returns:
        Tensor, final loss.
    """

    def __init__(self, config):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.smoothing = 0.1
        self.confidence = 0.9
        self.last_idx = (-1,)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.index_ids = Tensor(np.arange(config.batch_size * config.max_decode_length).reshape((-1, 1)), mstype.int32)
        self.gather_nd = P.GatherNd()
        self.expand = P.ExpandDims()
        self.concat = P.Concat(axis=-1)

    def construct(self, prediction_scores, label_ids, label_weights):
        """
        Construct network to calculate loss.

        Args:
            prediction_scores (Tensor): Prediction scores. [batchsize, seq_len, vocab_size]
            label_ids (Tensor): Labels. [batchsize, seq_len]
            label_weights (Tensor): Mask tensor. [batchsize, seq_len]

        Returns:
            Tensor, final loss.
        """
        prediction_scores = self.reshape(prediction_scores, (-1, self.vocab_size))
        label_ids = self.reshape(label_ids, (-1, 1))
        label_weights = self.reshape(label_weights, (-1,))
        tmp_gather_indices = self.concat((self.index_ids, label_ids))
        nll_loss = self.neg(self.gather_nd(prediction_scores, tmp_gather_indices))
        nll_loss = label_weights * nll_loss
        smooth_loss = self.neg(self.reduce_mean(prediction_scores, self.last_idx))
        smooth_loss = label_weights * smooth_loss
        loss = self.reduce_sum(self.confidence * nll_loss + self.smoothing * smooth_loss, ())
        loss = loss / self.batch_size
        return loss


class GNMTNetworkWithLoss(nn.Cell):
    """
    Provide  GNMT training loss through network.

    Args:
        config (BertConfig): The config of GNMT.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GNMTNetworkWithLoss, self).__init__()
        self.gnmt = GNMTTraining(config, is_training, use_one_hot_embeddings)
        self.loss = LabelSmoothedCrossEntropyCriterion(config)
        self.cast = P.Cast()

    def construct(self,
                  source_ids,
                  source_mask,
                  target_ids,
                  label_ids,
                  label_weights):
        prediction_scores = self.gnmt(source_ids, source_mask, target_ids)
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

        self.loss_scalar = P.ScalarSummary()

    def construct(self,
                  source_eos_ids,
                  source_eos_mask,
                  target_sos_ids,
                  target_eos_ids,
                  target_eos_mask,
                  sens=None):
        """
        Construct network.

        Args:
            source_eos_ids (Tensor): Source sentence.
            source_eos_mask (Tensor): Source padding mask.
            target_sos_ids (Tensor): Target sentence.
            target_eos_ids (Tensor): Prediction sentence.
            target_eos_mask (Tensor): Prediction padding mask.
            sens (Tensor): Loss sen.

        Returns:
            Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.
        """
        source_ids = source_eos_ids
        source_mask = source_eos_mask
        target_ids = target_sos_ids
        label_ids = target_eos_ids
        label_weights = target_eos_mask

        weights = self.weights
        loss = self.network(source_ids,
                            source_mask,
                            target_ids,
                            label_ids,
                            label_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # Alloc status.
        init = self.alloc_status()
        # Clear overflow buffer.
        init = F.depend(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(source_ids,
                                                 source_mask,
                                                 target_ids,
                                                 label_ids,
                                                 label_weights,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))

        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)
        if self.reducer_flag:
            # Apply grad reducer on grads.
            grads = self.grad_reducer(grads)
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
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
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)

        self.loss_scalar("loss", loss)

        ret = (loss, cond, scaling_sens)
        return F.depend(ret, succ)
