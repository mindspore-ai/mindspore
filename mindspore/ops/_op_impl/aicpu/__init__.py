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
from .init_data_set_queue import _init_data_set_queue_aicpu
from .dropout_genmask import _dropout_genmask_aicpu
from .get_next import _get_next_aicpu
from .print_tensor import _print_aicpu
from .topk import _top_k_aicpu
from .is_finite import _is_finite_aicpu
from .reshape import _reshape_aicpu
from .flatten import _flatten_aicpu
from .squeeze import _squeeze_aicpu
from .expand_dims import _expand_dims_aicpu
from .random_choice_with_mask import _random_choice_with_mask_aicpu
from .pack import _pack_aicpu
