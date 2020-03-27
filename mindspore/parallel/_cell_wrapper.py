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
"""Cell of auto parallel"""

from mindspore.nn.cell import Cell
from mindspore.ops.operations.comm_ops import AllGather


_allgather_cell = None


class AllGatherCell(Cell):
    """
    Allgather cell, used in model parallel scenario.
    To allgather the selected parameter slice from each device.
    """
    def __init__(self):
        super(AllGatherCell, self).__init__(auto_prefix=False)

        self.allgather = AllGather()

    def construct(self, x):
        x = self.allgather(x)

        return x


def get_allgather_cell():
    """Get AllGatherCell object."""
    global _allgather_cell
    if not _allgather_cell:
        _allgather_cell = AllGatherCell()

    return _allgather_cell


def destroy_allgather_cell():
    """Destroy AllGatherCell object."""
    global _allgather_cell
    if _allgather_cell:
        _allgather_cell = None
