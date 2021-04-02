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
"""less batch normalization"""
from ..cell import Cell

class LessBN(Cell):
    """
    Reduce the number of BN automatically to improve the network performance
    and ensure the network accuracy.

    Args:
        network (Cell): Network to be modified.

    Examples:
        >>> network = acc.LessBN(network)
    """

    def __init__(self, network):
        super(LessBN, self).__init__()
        self.network = network
        self.network.set_acc("less_bn")

    def construct(self, *inputs):
        return self.network(*inputs)
