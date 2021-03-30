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

"""
network config setting, gradient clip function and dynamic learning rate function
"""

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR


class GPTConfig:
    """
    GPT config class which defines the model size
    """
    def __init__(self,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=50257,
                 embedding_size=768,
                 num_layers=12,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        self.use_past = use_past




get_square_sum = C.MultitypeFuncGraph("get_square_sum")
@get_square_sum.register("Tensor")
def _get_square_sum(grad):
    norm = P.ReduceSum(False)(F.square(grad), ())
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad

class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """
    def __init__(self):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        square_sum = self.hyper_map(get_square_sum, grads)
        global_norms = F.sqrt(F.addn(square_sum) / F.scalar_to_array(len(square_sum)))
        return global_norms

class ClipByGlobalNorm(nn.Cell):
    """
    Clip grads by global norm
    """
    def __init__(self, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm()
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for GPT network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power=1.0, use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr
