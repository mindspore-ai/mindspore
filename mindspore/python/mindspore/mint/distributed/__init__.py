# Copyright 2024 Huawei Technologies Co., Ltd
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
Collective communication interface.

Note that the APIs in the following list need to preset communication environment variables.

For Ascend devices, it is recommended to use the msrun startup method
without any third-party or configuration file dependencies.
Please see the `msrun startup
<https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
for more details.
"""
from __future__ import absolute_import
from mindspore.mint.distributed.distributed import TCPStore
from mindspore.mint.distributed.distributed import (
    init_process_group,
    destroy_process_group,
    get_rank,
    get_world_size,
    new_group,
    get_backend,
    get_global_rank,
    get_process_group_ranks,
    get_group_rank,
    all_reduce,
    all_gather_into_tensor,
    all_gather_into_tensor_uneven,
    all_to_all,
    all_to_all_single,
    reduce_scatter_tensor,
    reduce_scatter_tensor_uneven,
    isend,
    irecv,
    send,
    recv,
    barrier,
    broadcast,
    reduce,
    P2POp,
    batch_isend_irecv,
    gather,
    scatter,
    all_gather,
    reduce_scatter,
    all_gather_object,
    broadcast_object_list,
    gather_object,
    scatter_object_list,
    is_available,
    is_initialized,
)

__all__ = [
    "init_process_group",
    "destroy_process_group",
    "get_rank",
    "get_world_size",
    "new_group",
    "get_backend",
    "get_global_rank",
    "get_process_group_ranks",
    "get_group_rank",
    "all_reduce",
    "all_gather_into_tensor",
    "all_gather_into_tensor_uneven",
    "all_to_all",
    "all_to_all_single",
    "reduce_scatter_tensor",
    "reduce_scatter_tensor_uneven",
    "isend",
    "irecv",
    "send",
    "recv",
    "gather",
    "scatter",
    "all_gather",
    "reduce_scatter",
    "barrier",
    "broadcast",
    "reduce",
    "P2POp",
    "batch_isend_irecv",
    "all_gather_object",
    "broadcast_object_list",
    "gather_object",
    "scatter_object_list",
    "is_available",
    "is_initialized",
]
