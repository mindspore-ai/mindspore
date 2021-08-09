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
# ============================================================================
"""Auto mixed precision."""
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype


class OutputTo(nn.Cell):
    "Cast cell output back to float16 or float32"

    def __init__(self, op, to_type=mstype.float16):
        super(OutputTo, self).__init__(auto_prefix=False)
        self._op = op
        validator.check_type_name('to_type', to_type, [mstype.float16, mstype.float32], None)
        self.to_type = to_type

    def construct(self, x):
        return F.cast(self._op(x), self.to_type)


def auto_mixed_precision(network):
    """Do keep batchnorm fp32."""
    cells = network.name_cells()
    change = False
    network.to_float(mstype.float16)
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif name == 'fc':
            network.insert_child_to_cell(name, OutputTo(subcell, mstype.float32))
            change = True
        elif isinstance(subcell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            network.insert_child_to_cell(name, OutputTo(subcell.to_float(mstype.float32), mstype.float16))
            change = True
        else:
            auto_mixed_precision(subcell)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())
