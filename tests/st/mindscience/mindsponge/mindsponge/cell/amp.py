# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""amp"""

import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import functional as F


class OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return F.cast(self._op(*x), mstype.float16)


# pylint: disable=W0212
def amp_convert(network, white_list=None):
    """Do keep cell fp32."""
    network.to_float(mstype.float16)
    if white_list is not None:
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            elif isinstance(subcell, white_list):
                network._cells[name] = OutputTo16(subcell.to_float(mstype.float32))
                change = True
            else:
                amp_convert(subcell, white_list)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())
