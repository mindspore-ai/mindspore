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
"""FastText for train"""
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.parameter import ParameterTuple
from mindspore.context import ParallelMode
from mindspore import nn
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore import context
from src.fasttext_model import FastText


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
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad

class FastTextNetWithLoss(nn.Cell):
    """
    Provide FastText training loss

    Args:
        vocab_size: vocabulary size
        embedding_dims: The size of each embedding vector
        num_class: number of labels
    """
    def __init__(self, vocab_size, embedding_dims, num_class):
        super(FastTextNetWithLoss, self).__init__()
        self.fasttext = FastText(vocab_size, embedding_dims, num_class)
        self.loss_func = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.squeeze = P.Squeeze(axis=1)
        self.print = P.Print()

    def construct(self, src_tokens, src_tokens_lengths, label_idx):
        """
        FastText network with loss.
        """
        predict_score = self.fasttext(src_tokens, src_tokens_lengths)
        label_idx = self.squeeze(label_idx)
        predict_score = self.loss_func(predict_score, label_idx)

        return predict_score


class FastTextTrainOneStepCell(nn.Cell):
    """
    Encapsulation class of fasttext network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(FastTextTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
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

        self.hyper_map = C.HyperMap()
        self.cast = P.Cast()

    def set_sens(self, value):
        self.sens = value

    def construct(self,
                  src_token_text,
                  src_tokens_text_length,
                  label_idx_tag):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(src_token_text,
                            src_tokens_text_length,
                            label_idx_tag)
        grads = self.grad(self.network, weights)(src_token_text,
                                                 src_tokens_text_length,
                                                 label_idx_tag,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        succ = self.optimizer(grads)
        return F.depend(loss, succ)
