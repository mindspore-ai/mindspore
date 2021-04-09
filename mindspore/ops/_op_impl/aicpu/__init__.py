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

"""aicpu ops"""
from .unique import _unique_aicpu
from .no_repeat_ngram import _no_repeat_ngram_aicpu
from .init_data_set_queue import _init_data_set_queue_aicpu
from .embedding_lookup import _embedding_lookup_aicpu
from .padding import _padding_aicpu
from .gather import _gather_aicpu
from .gather_grad import _gather_grad_aicpu
from .scatter import _scatter_aicpu
from .identity import _identity_aicpu
from .edit_distance import _edit_distance_aicpu
from .unique_with_pad import _unique_with_pad_aicpu
from .sub_and_filter import _sub_and_filter_aicpu
from .pad_and_shift import _pad_and_shift_aicpu
from .dropout_genmask import _dropout_genmask_aicpu
from .dropout2d import _dropout2d_aicpu
from .dropout3d import _dropout3d_aicpu
from .get_next import _get_next_aicpu
from .print_tensor import _print_aicpu
from .topk import _top_k_aicpu
from .is_finite import _is_finite_aicpu
from .reshape import _reshape_aicpu
from .flatten import _flatten_aicpu
from .squeeze import _squeeze_aicpu
from .expand_dims import _expand_dims_aicpu
from .randperm import _randperm_aicpu
from .random_choice_with_mask import _random_choice_with_mask_aicpu
from .stack import _stack_aicpu
from .uniform_candidate_sampler import _uniform_candidate_sampler_aicpu
from .log_uniform_candidate_sampler import _log_uniform_candidate_sampler_aicpu
from .compute_accidental_hits import _compute_accidental_hits_aicpu
from .ctcloss import _ctcloss_aicpu
from .reverse_sequence import _reverse_sequence_aicpu
from .crop_and_resize import _crop_and_resize_aicpu
from .rnnt_loss import _rnnt_loss_aicpu
from .random_categorical import _random_categorical_aicpu
from .cast import _cast_aicpu
from .mirror_pad import _mirror_pad_aicpu
from .mirror_pad_grad import _mirror_pad_grad_aicpu
from .standard_normal import _standard_normal_aicpu
from .gamma import _gamma_aicpu
from .poisson import _poisson_aicpu
from .update_cache import _update_cache_aicpu
from .cache_swap_table import _cache_swap_table_aicpu
from .uniform_int import _uniform_int_aicpu
from .uniform_real import _uniform_real_aicpu
from .standard_laplace import _standard_laplace_aicpu
from .strided_slice import _strided_slice_aicpu
from .strided_slice_grad import _strided_slice_grad_aicpu
from .end_of_sequence import _end_of_sequence_aicpu
from .fused_sparse_adam import _fused_sparse_adam_aicpu
from .fused_sparse_lazy_adam import _fused_sparse_lazy_adam_aicpu
from .fused_sparse_ftrl import _fused_sparse_ftrl_aicpu
from .fused_sparse_proximal_adagrad import _fused_sparse_proximal_adagrad_aicpu
from .meshgrid import _meshgrid_aicpu
from .trans_data import _trans_data_aicpu
from .stack_push_pop import _stack_init_aicpu
from .stack_push_pop import _stack_push_aicpu
from .stack_push_pop import _stack_pop_aicpu
from .stack_push_pop import _stack_destroy_aicpu
from .ctc_greedy_decoder import _ctc_greedy_decoder_aicpu
