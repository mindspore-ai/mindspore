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
"""api definition"""
import threading

class Hccl():
    """Hccl definition"""
    _instance_lock = threading.Lock()
    _instance = None
    _rank_id = 0
    _rank_size = 1
    _group_size = 4

    def __init__(self):
        pass

    # pylint: disable=unused-argument
    def __new__(cls, *args, **kwargs):
        if not hasattr(Hccl, "_instance") or Hccl._instance is None:
            with Hccl._instance_lock:
                if not hasattr(Hccl,
                               "_instance") or Hccl._instance is None:
                    Hccl._instance = object.__new__(cls)
                    Hccl._instance.__init__()
        return Hccl._instance

    @property
    def rank_id(self):
        return self._rank_id

    @rank_id.setter
    def rank_id(self, rank_id):
        self._rank_id = rank_id

    @property
    def rank_size(self):
        return self._rank_size

    @property
    def group_size(self):
        return self._group_size

    @rank_size.setter
    def rank_size(self, size):
        self._rank_size = size

# pylint: disable=unused-argument
def get_rank_id(group=None):
    hccl = Hccl()
    return hccl.rank_id


def get_rank_size(group=None):
    hccl = Hccl()
    if group is None:
        return hccl.rank_size
    if isinstance(group, str):
        return int(group.split("-")[0])
    raise ValueError

def get_group_size(group=None):
    hccl = Hccl()
    if group is None:
        return hccl.group_size
    if isinstance(group, str):
        return int(group.split("-")[0])
    raise ValueError

# pylint: disable=unused-argument
def get_world_rank_from_group_rank(group, group_rank_id):
    return group_rank_id

# pylint: disable=unused-argument
def get_group_rank_from_world_rank(world_rank_id, group):
    return world_rank_id

# pylint: disable=unused-argument
def create_group(group, rank_size, rank_ids):
    pass

# pylint: disable=unused-argument
def destroy_group(group):
    pass
