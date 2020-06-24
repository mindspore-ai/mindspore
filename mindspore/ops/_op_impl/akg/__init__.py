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

"""autodiff ops"""
from .abs import _abs_akg
from .add_n import _add_n_akg
from .add import _add_akg
from .apply_momentum import _apply_momentum_akg
from .assign import _assign_akg
from .inplace_assign import _inplace_assign_akg
from .assign_add import _assign_add_akg
from .bias_add_grad import _bias_add_grad_akg
from .bias_add import _bias_add_akg
from .cast import _cast_akg
from .clear_zero import _clear_zero_akg
from .conv_bn1 import _conv_bn1_akg
from .conv2d_backprop_filter import _conv2d_backprop_filter_akg
from .conv2d_backprop_input import _conv2d_backprop_input_akg
from .conv2d import _conv2d_akg
from .div import _div_akg
from .equal_count import _equal_count_akg
from .exp import _exp_akg
from .five2four import _five2four_akg
from .four2five import _four2five_akg
from .fused_batch_norm_grad import _fused_batch_norm_grad_akg
from .fused_batch_norm_infer import _fused_batch_norm_infer_akg
from .fused_batch_norm import _fused_batch_norm_akg
from .fused_bn1_grad import _bn1_grad_akg
from .fused_bn1 import _fused_bn1_akg
from .fused_bn2_grad import _bn2_grad_akg
from .fused_bn2 import _fused_bn2_akg
from .fused_bn3_grad import _bn3_grad_akg
from .fused_bn3 import _fused_bn3_akg
from .gather_v2 import _gather_v2_akg
from .less import _less_akg
from .log import _log_akg
from .matmul import _matmul_akg
from .max_pool_grad_with_argmax import _max_pool_grad_with_argmax_akg
from .max_pool_with_argmax import _max_pool_with_argmax_akg
from .max import _max_akg
from .maximum import _maximum_akg
from .mean_grad import _mean_grad_akg
from .mean import _mean_akg
from .minimum import _minimum_akg
from .mul import _mul_akg
from .neg import _neg_akg
from .one_hot import _one_hot_akg
from .pow import _power_akg
from .real_div import _real_div_akg
from .reciprocal import _reciprocal_akg
from .reduce_max import _reduce_max_akg
from .reduce_mean import _reduce_mean_akg
from .reduce_sum import _reduce_sum_akg
from .relu_grad import _relu_grad_akg
from .relu import _relu_akg
from .reshape import _reshape_akg
from .round import _round_akg
from .rsqrt import _rsqrt_akg
from .select import _select_akg
from .softmax import _softmax_akg
from .sparse_softmax_cross_entropy_with_logits import _sparse_softmax_cross_entropy_with_logits_akg
from .sqrt import _sqrt_akg
from .strided_slice import _strided_slice_akg
from .sub import _sub_akg
from .sum import _sum_akg
from .tile import _tile_akg
from .zeros_like import _zeros_like_akg
from .argmax import _argmax_akg
from .floordiv import _floor_div_akg
from .equal import _equal_akg
from .greater_equal import _greater_equal_akg
from .less_equal import _less_equal_akg
from .expand_dims import _expand_dims_akg
from .greater import _greater_akg
from .equiv_format import _equiv_format_akg
from . import gpu
