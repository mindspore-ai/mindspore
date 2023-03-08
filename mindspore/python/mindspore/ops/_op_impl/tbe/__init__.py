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

"""tbe ops"""
from .broadcast_to import _broadcast_to_tbe  # The name is occupied
from .broadcast_to_ds import _broadcast_to_ds_tbe  # The name is occupied
from .batch_to_space import _batch_to_space_tbe  # attr type is listInt，not listListInt
from .batch_to_space_nd import _batch_to_space_nd_tbe  # attr type is listInt，not listListInt
from .batch_to_space_nd_v2 import _batch_to_space_nd_v2_tbe  # The name is occupied
from .space_to_batch import _space_to_batch_tbe  # attr type is listInt，not listListInt
from .space_to_batch_nd import _space_to_batch_nd_tbe  # attr type is listInt，not listListInt
from .dynamic_gru_v2 import _dynamic_gru_v2_tbe  # input4 is None, GE will change to hidden op by pass
from .dynamic_rnn import _dynamic_rnn_tbe  # input4 is None, GE will change to hidden op by pass
from .kl_div_loss_grad import _kl_div_loss_grad_tbe  # Accuracy issues
from .inplace_index_add import _inplace_index_add_tbe  # check support failed if var has only one dimension
from .scatter_nd_update_ds import _scatter_nd_update_ds_tbe  # not support int8 in op json
from .scatter_nd_add import _scatter_nd_add_tbe  # not support int8 in op json
from .cast_ds import _cast_ds_tbe  # Accuracy issues
from .avg_pool_3d_grad import _avg_pool_3d_grad_tbe  # Second device format is facz_3d, but in json, the key is ndhwc
from .data_format_dim_map_ds import _data_format_dim_map_ds_tbe  # attr order swap
from .depthwise_conv2d import _depthwise_conv2d_tbe  # Accuracy issues(second format is error in python)
from .acos import _acos_tbe # Accuracy issues(task error in parallel)
from .trans_data_ds import _trans_data_ds_tbe # support bool
from .scatter_nd_d import _scatter_nd_d_tbe # in python no check supported
from .assign_add_ds import _assign_add_ds_tbe # "Frac_nz in pangu not support"
from .atomic_addr_clean import _atomic_addr_clean_tbe # need to clean addr larger than 2G, int32 is not enough
from .assign import _assign_tbe # Different formats of assign inputs cause memory to increase
from .npu_clear_float_status_v2 import _npu_clear_float_status_v2_tbe  # io mismatch
from .npu_get_float_status_v2 import _npu_get_float_status_v2_tbe  # io mismatch
