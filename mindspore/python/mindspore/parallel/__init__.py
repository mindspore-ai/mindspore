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
"""Interfaces for parallel-related functionality"""
from __future__ import absolute_import

from mindspore.parallel.algo_parameter_config import get_algo_parameters, reset_algo_parameters, \
    set_algo_parameters
from mindspore.parallel.checkpoint_transform import rank_list_for_transform, transform_checkpoint_by_rank, \
    transform_checkpoints, merge_pipeline_strategys
from mindspore.parallel.shard import shard

__all__ = ["set_algo_parameters", "reset_algo_parameters", "get_algo_parameters", "rank_list_for_transform",
           "transform_checkpoint_by_rank", "transform_checkpoints", "merge_pipeline_strategys", "shard"]
