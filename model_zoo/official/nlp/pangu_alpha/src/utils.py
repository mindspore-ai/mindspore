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
from mindspore.train.model import Model as MsModel
from mindspore.parallel._utils import _get_global_rank
from mindspore.communication.management import get_group_size
from mindspore.nn import AdamWeightDecay
from mindspore.common import Parameter, ParameterTuple
from mindspore.common.initializer import initializer


class FP32StateAdamWeightDecay(AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training PanGu-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       eps=eps,
                                                       weight_decay=weight_decay)

        self.moments1 = self.clone_state(self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = self.clone_state(self.parameters, prefix='adam_v', init='zeros')

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


class Model(MsModel):
    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        super(Model, self).__init__(network, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
                                    eval_network=eval_network,
                                    eval_indexes=eval_indexes, amp_level=amp_level, **kwargs)
        # In avoid of the pylint
        self.init = self._init

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
        # Square sum of gradients for current rank
        square_sum_dp = self.hyper_map(get_square_sum, grads, self.values)
        # Global square sum of gradients
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
    Warmup-decay learning rate for PanguAlpha network.
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

def add_inference_params(opt):
    """Add inference params"""
    opt.add_argument("--frequency_penalty",
                     type=float,
                     default=1.5,
                     help="coefficient for frequency_penalty")
    opt.add_argument("--presence_penalty",
                     type=float,
                     default=0.3,
                     help="coefficient for presence_penalty")
    opt.add_argument("--max_generate_length",
                     type=int,
                     default=500,
                     help="the maximum number of generated token")
    opt.add_argument("--top_k_num",
                     type=int,
                     default=3,
                     help="the number for top_k sampling")
    opt.add_argument("--top_p",
                     type=float,
                     default=1.0,
                     help="top_p sampling threshold, enabled if less than 1.0")
    opt.add_argument("--end_token",
                     type=int,
                     default=9,
                     help="the token id for <end of document>")

def add_training_params(opt):
    """Add training params"""
    opt.add_argument("--seq_length",
                     type=int,
                     default=1024,
                     help="sequence length, default is 1024.")
    opt.add_argument("--vocab_size",
                     type=int,
                     default=40000,
                     help="vocabulary size, default is 40000.")
    opt.add_argument("--embedding_size",
                     type=int,
                     default=16384,
                     help="embedding table size, default is 16384.")
    opt.add_argument("--num_layers",
                     type=int,
                     default=64,
                     help="total layers, default is 64.")
    opt.add_argument("--num_heads",
                     type=int,
                     default=128,
                     help="head size, default is 128.")
    opt.add_argument("--stage_num",
                     type=int,
                     default=4,
                     help="Pipeline stage num, default is 4.")
    opt.add_argument("--micro_size",
                     type=int,
                     default=1,
                     help="Pipeline micro_size, default is 1.")
    opt.add_argument("--eod_reset",
                     type=int,
                     default=1,
                     help="Enable eod mask, default is 1.")
    opt.add_argument("--warmup_step",
                     type=int,
                     default=2000,
                     help="Warmup step, default is 2000.")
    opt.add_argument("--optimizer",
                     type=str,
                     default="adam",
                     choices=["adam", "lamb"],
                     help="select which optimizer to be used, default adam")
    opt.add_argument("--eod_id",
                     type=int,
                     default=6,
                     help="The id of end of document")
    opt.add_argument("--epoch_size",
                     type=int,
                     default=1,
                     help="The training epoch")
    opt.add_argument("--sink_size",
                     type=int,
                     default=2,
                     help="The sink size of the training")

def get_args(inference=False):
    """train function for PanguAlpha"""
    parser = argparse.ArgumentParser(description="PanguAlpha training")
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
    parser.add_argument("--load_ckpt_name",
                        type=str,
                        default='PANGUALPHA3.ckpt',
                        help="checkpint file name.")
    parser.add_argument("--load_ckpt_path",
                        type=str,
                        default=None,
                        help="predict file path.")
    parser.add_argument('--data_url',
                        required=False,
                        default=None,
                        help='Location of data.')
    parser.add_argument('--train_url',
                        required=False,
                        default=None,
                        help='Location of training outputs.')
    parser.add_argument("--run_type",
                        type=str,
                        default="predict",
                        choices=["train", "predict"],
                        help="The run type")
    parser.add_argument("--mode",
                        type=str,
                        default="2.6B",
                        choices=["200B", "13B", "2.6B", "self_define"],
                        help="The train/eval mode")
    parser.add_argument("--strategy_load_ckpt_path",
                        type=str,
                        default="",
                        help="The training prallel strategy for the model.")
    parser.add_argument("--tokenizer_path",
                        type=str,
                        default="./tokenizer_path",
                        help="The path where stores vocab and vocab model file")
    parser.add_argument("--param_init_type",
                        type=str,
                        default="fp32",
                        help="The initialization type for parameters. Default fp32.")
    parser.add_argument("--incremental_training",
                        type=int,
                        default=0,
                        help="The incremental training. Default 1.")
    add_training_params(parser)
    if inference:
        add_inference_params(parser)
    args_opt = parser.parse_args()

    return args_opt
