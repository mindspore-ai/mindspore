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
"""expanders init"""

from .assign_add import AssignAdd
from .bias_add import BiasAdd
from .bias_add_grad import BiasAddGrad
from .clip_by_norm_no_div_sum import ClipByNormNoDivSum
from .dropout_grad import DropoutGrad
from .fused_adam import FusedAdam
from .fused_adam_weight_decay import FusedAdamWeightDecay
from .gelu import GeLU
from .gelu_grad import GeLUGrad
from .gkdropout import GkDropout
from .layernorm import LayerNorm
from .layernorm_grad import LayerNormGrad
from .logsoftmax import LogSoftmax
from .logsoftmax_grad import LogSoftmaxGrad
from .maximum_grad import MaximumGrad
from .minimum_grad import MinimumGrad
from .reduce_mean import ReduceMean
from .softmax import Softmax
from .sigmoid import Sigmoid
from .sigmoid_grad import SigmoidGrad
from .sigmoid_cross_entropy_with_logits import SigmoidCrossEntropyWithLogits
from .sigmoid_cross_entropy_with_logits_grad import SigmoidCrossEntropyWithLogitsGrad
from .softmax_cross_entropy_with_logits import SoftmaxCrossEntropyWithLogits
from .sqrt_grad import SqrtGrad
from .square import Square
from .tanh_grad import TanhGrad
from .tile import Tile
from .lamb_apply_optimizer_assign import LambApplyOptimizerAssign
