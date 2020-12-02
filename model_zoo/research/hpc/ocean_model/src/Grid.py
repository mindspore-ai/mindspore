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
"""Grid"""

import mindspore.nn as nn
from src.stencil import AXB, AYB, AZB


class Grid(nn.Cell):
    """
    init C grid
    """
    def __init__(self, im, jm, km, stencil_width=1):
        super(Grid, self).__init__()
        self.im = im
        self.jm = jm
        self.km = km
        self.x_map = [1, 0, 3, 2, 5, 4, 7, 6]
        self.y_map = [2, 3, 0, 1, 6, 7, 4, 5]
        self.z_map = [4, 5, 6, 7, 0, 1, 2, 3]
        self.AXB = AXB(stencil_width=stencil_width)
        self.AYB = AYB(stencil_width=stencil_width)
        self.AZB = AZB(stencil_width=stencil_width)

    def construct(self, dx, dy, dz):
        """construct"""
        dx0 = self.AYB(self.AXB(dx))
        dy0 = self.AYB(self.AXB(dy))
        dz0 = dz

        dx1 = self.AYB(dx)
        dy1 = self.AYB(dy)
        dz1 = self.AZB(dz)

        dx2 = self.AXB(dx)
        dy2 = self.AXB(dy)

        dx3 = dx
        dy3 = dy

        x_d = (dx0, dx1, dx2, dx3)
        y_d = (dy0, dy1, dy2, dy3)
        z_d = (dz0, dz1)

        return x_d, y_d, z_d
