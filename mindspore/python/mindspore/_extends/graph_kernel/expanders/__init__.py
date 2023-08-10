# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""expanders init. Deprecated, please add the new operators in the c++ file"""


from .complex import CAbs, CAdd, CDiv, CMul, CSub, CRealDiv
from .equal_count import EqualCount
from .fused_adam import FusedAdam
from .fused_adam_weight_decay import FusedAdamWeightDecay
from .reduce_mean import ReduceMean
from .sigmoid_cross_entropy_with_logits import SigmoidCrossEntropyWithLogits
from .sigmoid_grad import SigmoidGrad
from .softmax_cross_entropy_with_logits import SoftmaxCrossEntropyWithLogits
from .softmax_grad_ext import SoftmaxGradExt
from .sqrt_grad import SqrtGrad
from .squared_difference import SquaredDifference
from .square_sum_v1 import SquareSumV1
from .square_sum_all import SquareSumAll
from .tanh_grad import TanhGrad
from .softsign import Softsign
