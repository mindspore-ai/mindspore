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
"""mdnn class"""

import numpy as np
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype


class Mdnn(nn.Cell):
    """Mdnn"""

    def __init__(self, dim=258, dr=0.5):
        super(Mdnn, self).__init__()
        self.dim = dim
        self.dr = dr  # dropout_ratio
        self.fc1 = nn.Dense(dim, 512)
        self.fc2 = nn.Dense(512, 512)
        self.fc3 = nn.Dense(512, 512)
        self.fc4 = nn.Dense(512, 129)
        self.tanh = nn.Tanh()

    def construct(self, x):
        """construct"""
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class TransCrdToCV(nn.Cell):
    """TransCrdToCV"""

    def __init__(self, simulation):
        super(TransCrdToCV, self).__init__()
        self.atom_numbers = simulation.atom_numbers
        self.transfercrd = P.TransferCrd(0, 129, 129, self.atom_numbers)
        self.box = Tensor(simulation.box_length)
        self.radial = Parameter(Tensor(np.zeros([129,]), mstype.float32))
        self.angular = Parameter(Tensor(np.zeros([129,]), mstype.float32))
        self.output = Parameter(Tensor(np.zeros([1, 258]), mstype.float32))
        self.charge = simulation.charge

    def updatecharge(self, t_charge):
        """update charge in simulation"""
        self.charge[:129] = t_charge[0] * 18.2223
        return self.charge

    def construct(self, crd, last_crd):
        """construct"""
        self.radial, self.angular, _, _ = self.transfercrd(crd, last_crd, self.box)
        self.output = P.Concat()((self.radial, self.angular))
        self.output = P.ExpandDims()(self.output, 0)
        return self.output
