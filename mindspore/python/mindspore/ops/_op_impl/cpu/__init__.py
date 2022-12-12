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

"""cpu ops"""
from .cast import _cast_cpu
from .mul import _mul_cpu
from .sub import _sub_cpu
from .pow import _pow_cpu
from .real_div import _real_div_cpu
from .div import _div_cpu
from .concat import _concat_cpu
from .concat_offset import _concat_offset_cpu
from .dynamic_shape import _dynamic_shape_cpu
from .dynamic_stitch import _dynamic_stitch_cpu
from .split import _split_cpu
from .adam import _adam_cpu
from .adam_weight_decay import _adam_weight_decay_cpu
from .arg_max import _arg_max_cpu
from .arg_min_with_value import _arg_min_with_value_cpu
from .arg_max_with_value import _arg_max_with_value_cpu
from .dropout import _dropout_cpu
from .dropout_grad import _dropout_grad_cpu
from .gather_d import _gather_cpu
from .tensor_shape import _tensor_shape_cpu
from .gather_d_grad import _gather_d_grad_cpu
from .gather_d_grad_v2 import _gather_d_grad_v2_cpu
from .gather_v2 import _gather_v2_cpu
from .maximum import _maximum_cpu
from .maximum_grad import _maximum_grad_cpu
from .conv2d import _conv2d_cpu
from .conv3d import _conv3d_cpu
from .hsigmoid import _hsigmoid_cpu
from .hsigmoid_grad import _hsigmoid_grad_cpu
from .hswish import _hswish_cpu
from .hswish_grad import _hswish_grad_cpu
from .identity_n import _identity_n_cpu
from .is_finite import _is_finite_cpu
from .layer_norm import _layer_norm_cpu
from .layer_norm_grad import _layer_norm_grad_cpu
from .minimum import _minimum_cpu
from .minimum_grad import _minimum_grad_cpu
from .equal_count import _equal_count_cpu
from .mirror_pad import _mirror_pad_cpu
from .mirror_pad_grad import _mirror_pad_grad_cpu
from .stack import _stack_cpu
from .reduce_mean import _reduce_mean_cpu
from .reduce_max import _reduce_max_cpu
from .reduce_sum import _reduce_sum_cpu
from .reduce_min import _reduce_min_cpu
from .reduce_prod import _reduce_prod_cpu
from .reduce_all import _reduce_all_cpu
from .reduce_any import _reduce_any_cpu
from .transpose import _transpose_cpu
from .tile import _tile_cpu
from .top_k import _top_k_cpu
from .one_hot import _one_hot_cpu
from .pad import _pad_cpu
from .range import _range_cpu
from .tensor_copy_slices import _tensor_copy_slices_cpu
from .l2loss import _l2loss_cpu
from .pyexecute import _pyexecute_cpu
from .pyfunc import _pyfunc_cpu
from .buffer_append import _buffer_append_cpu
from .buffer_get import _buffer_get_cpu
from .buffer_sample import _buffer_sample_cpu
from .priority_replay_buffer import _prb_push_op_cpu
from .priority_replay_buffer import _prb_sample_op_cpu
from .space_to_batch_nd import _space_to_batch_nd_cpu
from .sspaddmm import _sspaddmm_cpu
