# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""GPT-2 finetune for downstream task"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size

from .utils.CrossEntropy import CrossEntropyCalculationWithMask
from .clip_grad_utils import clip_grad
from .GPT2ForLanguageModel import GPT2LanguageModel
from .GPT2ForLambada import GPT2LambadaModel
from .GPT2ForCBT import GPT2CBTModel
from .GPT2ForTranslation import GPT2TranslationModel
from .GPT2ForReadComprehension import GPT2CoQAModel
from .GPT2ForSummarization import GPT2SummarizationModel


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class GPT2FinetuneCell(nn.Cell):
    """
    Specifically defined for finetuning where only three inputs tensor are needed.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(GPT2FinetuneCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
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
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    def construct(self,
                  input_ids,
                  input_mask,
                  label_ids,
                  sens=None):
        """
        GPT2 Finetune.

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary
        """

        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        if not self.gpu_target:
            init = self.alloc_status()
            init = F.depend(init, loss)
            clear_before_grad = self.clear_before_grad(init)
            scaling_sens = F.depend(scaling_sens, clear_before_grad)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            init = F.depend(init, grads)
            flag = self.get_status(init)
            init = F.depend(init, flag)
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class GPT2LM(nn.Cell):
    """
    Train interface for Language Modeling finetuning task.

    Args:
        config (class): the configuration of GPT-2 model.
        is_training (bool): whether to train.
        use_one_hot_embeddings (bool): whether to use onehot embeddings.
    """

    def __init__(self, config=None, is_training=None, use_one_hot_embeddings=False):
        super(GPT2LM, self).__init__()
        self.gpt2 = GPT2LanguageModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training,
                                                    num_labels=self.num_labels,
                                                    config=config)
        self.is_training = is_training
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask, label_ids):
        """
        construct function for Language Modeling

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary

        Returns:
            lm_logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """
        lm_logits = self.gpt2(input_ids, input_mask)  # [batch_size, seq_length, vocab_size]

        if self.is_training:
            shift_logits = lm_logits[::, :-1, ::]  # [batch_size, seq_length - 1, vocab_size]
            shift_logits = self.reshape(shift_logits, (-1, self.num_labels))  # [batch * (seq_length - 1), vocab_size]
            label_ids = label_ids[::, 1:]
            input_mask = input_mask[::, 1:]

            loss = self.loss(shift_logits, label_ids, input_mask)
            return loss

        return lm_logits


class GPT2Lambada(nn.Cell):
    """
    Train interface for Lambada finetuning task.

    Args:
        config (class): the configuration of GPT-2 model.
        is_training (bool): whether to train.
        use_one_hot_embeddings (bool): whether to use onehot embeddings.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2Lambada, self).__init__()
        self.gpt2 = GPT2LambadaModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training,
                                                    num_labels=self.num_labels,
                                                    config=config)
        self.is_training = is_training
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask, label_ids=None):
        """
        construct function for Lambada task

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.

        Returns:
            lm_logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """
        lm_logits = self.gpt2(input_ids, input_mask)  # [batch_size, seq_length, vocab_size]

        if self.is_training:
            shift_logits = lm_logits[:, :-1, :]  # [batch_size, seq_length - 1, vocab_size]
            shift_logits = self.reshape(shift_logits, (-1, self.num_labels))  # [batch * (seq_length - 1), vocab_size]
            label_ids = label_ids[::, 1:]
            input_mask = input_mask[::, 1:]

            loss = self.loss(shift_logits, label_ids, input_mask)

            return loss

        return lm_logits


class GPT2CBT(nn.Cell):
    """
    Train interface for Children's Book Test finetuning task.

    Args:
        config (class): the configuration of GPT-2 model.
        is_training (bool): whether to train.
        use_one_hot_embeddings (bool): whether to use onehot embeddings.
    """

    def __init__(self, config=None, is_training=None, use_one_hot_embeddings=False):
        super(GPT2CBT, self).__init__()
        self.gpt2 = GPT2CBTModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training,
                                                    num_labels=self.num_labels,
                                                    config=config)
        self.is_training = is_training
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask):
        """
        construct function for CBT task

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.

        Returns:
            lm_logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """
        lm_logits = self.gpt2(input_ids, input_mask)  # [batch_size, seq_length, vocab_size]

        if self.is_training:
            shift_logits = lm_logits[::, :-1, ::]  # [batch_size, seq_length - 1, vocab_size]
            shift_logits = self.reshape(shift_logits, (-1, self.num_labels))  # [batch * (seq_length - 1), vocab_size]
            label_ids = input_ids[::, 1:]
            input_mask = input_mask[::, 1:]

            loss = self.loss(shift_logits, label_ids, input_mask)
            return loss

        return lm_logits


class GPT2Translation(nn.Cell):
    """
    Train interface for Translation finetuning task.

    Args:
        config (class): the configuration of GPT-2 model.
        is_training (bool): whether to train.
        use_one_hot_embeddings (bool): whether to use onehot embeddings.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2Translation, self).__init__()
        self.gpt2 = GPT2TranslationModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training,
                                                    num_labels=self.num_labels,
                                                    config=config)
        self.is_training = is_training
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_ids, input_mask, label_ids):
        """
        construct function for Translation task

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary

        Returns:
            translation_logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """
        translation_logits = self.gpt2(input_ids, input_mask)  # [batch_size, seq_length, vocab_size]
        translation_logits = self.log_softmax(translation_logits)

        if self.is_training:
            shift_logits = translation_logits[::, :-1, ::]  # [batch_size, seq_length - 1, vocab_size]
            shift_logits = self.reshape(shift_logits, (-1, self.num_labels))  # [batch * (seq_length - 1), vocab_size]
            label_ids = label_ids[::, 1:]
            input_mask = input_mask[::, 1:]

            loss = self.loss(shift_logits, label_ids, input_mask)
            return loss

        return translation_logits


class GPT2Summarization(nn.Cell):
    """
    Train interface for Summary finetuning task.

    Args:
        config (class): the configuration of GPT-2 model.
        is_training (bool): whether to train.
        use_one_hot_embeddings (bool): whether to use onehot embeddings.
    """

    def __init__(self, config=None, is_training=None, use_one_hot_embeddings=False):
        super(GPT2Summarization, self).__init__()
        self.gpt2 = GPT2SummarizationModel(config, is_training, use_one_hot_embeddings)
        self.is_training = is_training
        self.last_idx = (-1,)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.cast = P.Cast()
        self.loss_function = CrossEntropyCalculationWithMask(num_labels=self.vocab_size,
                                                             is_training=self.is_training,
                                                             config=config)

    def construct(self, input_ids, input_mask, label_ids):
        """
        construct function for Language Modeling

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary

        Returns:
            loss (mstype.float32): if is_training is True, return the computed loss.
        """
        output = self.gpt2(input_ids, input_mask)

        shift_logits = output[::, :-1, ::]
        shift_logits = self.reshape(shift_logits, (-1, self.vocab_size))
        shift_logits = self.log_softmax(shift_logits)
        label_ids = label_ids[::, 1:]
        input_mask = input_mask[::, 1:]

        loss = self.loss_function(shift_logits, label_ids, input_mask)

        return loss


class GPT2CoQA(nn.Cell):
    """
    Train interface for Reading Comprehension finetuning task.

    Args:
        config (class): the configuration of GPT-2 model.
        is_training (bool): whether to train.
        use_one_hot_embeddings (bool): whether to use onehot embeddings.
    """
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2CoQA, self).__init__()
        self.gpt2 = GPT2CoQAModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training,
                                                    num_labels=self.num_labels,
                                                    config=config)
        self.is_training = is_training
        self.reshape = P.Reshape()
        self.log_softmax = P.LogSoftmax(axis=-1)

    def construct(self, input_ids, input_mask, label_ids=None):
        """
        construct function for reading comprehension task

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sequence padding mask, where 0 indicates padding position.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary

        Returns:
            lm_logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """
        lm_logits = self.gpt2(input_ids, input_mask)
        lm_logits = self.log_softmax(lm_logits)

        if self.is_training:
            shift_logits = lm_logits[::, :-1, ::]
            shift_logits = self.reshape(shift_logits, (-1, self.num_labels))
            label_ids = label_ids[::, 1:]
            input_mask = input_mask[::, 1:]

            loss = self.loss(shift_logits, label_ids, input_mask)
            return loss

        return lm_logits
