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
"""
Collective communication interface.

Note that the APIs in the following list need to preset communication environment variables.

For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
Please see the `rank table Startup
<https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html>`_
for more details.

For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun Startup
<https://www.mindspore.cn/tutorials/experts/en/master/parallel/mpirun.html>`_ .

For the CPU device, users need to write a dynamic cluster startup script, please see the `Dynamic Cluster Startup
<https://www.mindspore.cn/tutorials/experts/en/master/parallel/dynamic_cluster.html>`_ .
"""

from mindspore.communication.management import GlobalComm, init, release, get_rank, \
    get_group_size, get_world_rank_from_group_rank, \
    get_group_rank_from_world_rank, create_group, HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, \
    MCCL_WORLD_COMM_GROUP, get_local_rank, get_local_rank_size, destroy_group


__all__ = [
    "GlobalComm", "init", "release", "get_rank", "get_group_size", "get_world_rank_from_group_rank",
    "get_group_rank_from_world_rank", "create_group", "HCCL_WORLD_COMM_GROUP", "NCCL_WORLD_COMM_GROUP",
    "MCCL_WORLD_COMM_GROUP", "get_local_rank", "get_local_rank_size", "destroy_group"
]
