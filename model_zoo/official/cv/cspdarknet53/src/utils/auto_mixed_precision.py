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
"""auto mixed precision"""
from collections.abc import Iterable
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


def check_type_name(arg_name, arg_type, valid_types, prim_name):
    """Checks whether a type in some specified types"""
    valid_types = valid_types if isinstance(valid_types, Iterable) else (valid_types,)

    def raise_error_msg():
        """func for raising error message when check failed"""
        type_names = [t.__name__ if hasattr(t, '__name__') else t for t in valid_types]
        num_types = len(valid_types)
        msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
        raise TypeError(f"{msg_prefix} '{arg_name}' should be {'one of ' if num_types > 1 else ''}"
                        f"{type_names if num_types > 1 else type_names[0]}, "
                        f"but got {arg_type.__name__ if hasattr(arg_type, '__name__') else repr(arg_type)}.")

    if isinstance(arg_type, type(mstype.tensor)):
        arg_type = arg_type.element_type()
    if arg_type not in valid_types:
        raise_error_msg()
    return arg_type

class OutputTo(nn.Cell):
    """Cast cell output back to float16 or float32"""

    def __init__(self, op, to_type=mstype.float16):
        super(OutputTo, self).__init__(auto_prefix=False)
        self._op = op
        check_type_name('to_type', to_type, [mstype.float16, mstype.float32], None)
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
        else:
            auto_mixed_precision(subcell)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())
