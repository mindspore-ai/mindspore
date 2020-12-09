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
"""expanders init"""

from .gelu import expand_gelu
from .gelu_grad import expand_gelugrad
from .layernorm import expand_layernorm
from .softmax import expand_softmax
from .square import expand_square
from .bias_add import expand_biasadd
from .bias_add_grad import expand_biasaddgrad
from .fused_adam import expand_fusedadam
from .fused_adam_weight_decay import expand_fusedadamweightdecay
from .reduce_mean import expand_reducemean
from .tanh_grad import expand_tanhgrad
from .maximum_grad import expand_maximumgrad
from .minimum_grad import expand_minimumgrad
from .dropout_grad import expand_dropoutgrad
from .layernorm_grad import expand_layernormgrad
from .logsoftmax import expand_logsoftmax
from .logsoftmax_grad import expand_logsoftmaxgrad
from .gkdropout import expand_gkdropout
from .tile import expand_tile
from .sqrt_grad import expand_sqrtgrad
from .clip_by_norm_no_div_sum import expand_clipbynormnodivsum
