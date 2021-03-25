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
"""
network config setting, gradient clip function and dynamic learning rate function
"""
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR

from mindspore.parallel._utils import _get_global_rank
from mindspore.communication.management import get_group_size

class PanGu1Config:
    """
    PanGu1 config class which defines the model size
    """
    def __init__(self,
                 data_parallel_num,
                 model_parallel_num,
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
                 use_past=False,
                 self_layernorm=True,
                 forward_reduce_scatter=True,
                 word_emb_dp=True,
                 stage_num=16,
                 micro_size=32,
                 eod_reset=False,
                 use_top_query_attention=False,
                 use_recompute=True):
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
        self.dp = data_parallel_num
        self.mp = model_parallel_num
        self.self_layernorm = self_layernorm
        self.forward_reduce_scatter = forward_reduce_scatter
        self.stage_num = stage_num
        self.micro_size = micro_size
        self.word_emb_dp = word_emb_dp
        self.eod_reset = eod_reset
        self.use_recompute = use_recompute
        self.use_top_query_attention = use_top_query_attention

    def __str__(self):
        info = "[PanGu1 Config]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 10
        return info


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Tensor")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad) / value, ())
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
    def __init__(self, params):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.allreduce_filter = tuple(
            "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table"
            not in x.name for x in params)
        self.length = len(params)
        self.values = []
        self.group_size = get_group_size()
        for item in self.allreduce_filter:
            if item:
                self.values.append(Tensor([1.0], mstype.float32))
            else:
                self.values.append(Tensor([self.group_size*1.0], mstype.float32))
        self.values = tuple(self.values)
    def construct(self, grads):
        square_sum_dp = self.hyper_map(get_square_sum, grads, self.values)
        global_norms = F.sqrt(P.AllReduce()(F.addn(square_sum_dp)))
        return global_norms


class ClipByGlobalNorm(nn.Cell):
    """

    Clip grads by global norm

    """
    def __init__(self, params, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads, global_norm_value


def _get_model_parallel_group(dp, mp):
    rank = _get_global_rank()
    group = range(0, mp)
    index = rank // dp
    return [x + index * mp for x in group]



class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PanGu1 network.
    """
    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True,
                 lr_scale=0.125):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine
        self.lr_scale = lr_scale

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr * self.lr_scale


def get_args():
    """train function for PanGu1"""
    parser = argparse.ArgumentParser(description="PanGu1 training")
    parser.add_argument('--device_id',
                        type=int,
                        default=0,
                        help="Device id, default is 0.")
    parser.add_argument("--device_num",
                        type=int,
                        default=128,
                        help="Use device nums, default is 1.")
    parser.add_argument("--distribute",
                        type=str,
                        default="true",
                        choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adam",
                        choices=["adam", "lamb"],
                        help="select which optimizer to be used, default adam")
    parser.add_argument("--epoch_size",
                        type=int,
                        default=1,
                        help="Epoch size, default is 10.")
    parser.add_argument("--warmup_step",
                        type=int,
                        default=2000,
                        help="Warmup step, default is 10000.")
    parser.add_argument("--start_lr",
                        type=float,
                        default="1e-4",
                        help="Start learning rate, default is 1e-4.")
    parser.add_argument("--end_lr",
                        type=float,
                        default="1e-6",
                        help="End learning rate, default is 1e-6.")
    parser.add_argument("--sink_size",
                        type=int,
                        default=1,
                        help="Sink size for every iteration, default is 1")
    parser.add_argument('--data_url',
                        required=True,
                        default=None,
                        help='Location of data.')
    parser.add_argument('--whl_pkg',
                        type=str,
                        default='',
                        help='Location of mindspore whl.')
    parser.add_argument("--sample_count",
                        type=int,
                        default=1000000,
                        help="sample_count, default is 1000000.")
    parser.add_argument("--eod_id",
                        type=int,
                        default=9,
                        help="eod_id.")
    parser.add_argument("--eod_reset",
                        type=int,
                        default=1,
                        help="eod_reset 0/1.")
    parser.add_argument("--full_batch",
                        type=int,
                        default=0,
                        help="full_batch 0/1.")
    parser.add_argument("--per_batch_size",
                        type=int,
                        default=1,
                        help="The batch size of each card.")
    parser.add_argument("--mp",
                        type=int,
                        default=8,
                        help="The model parallel number.")

    args_opt = parser.parse_args()

    return args_opt
