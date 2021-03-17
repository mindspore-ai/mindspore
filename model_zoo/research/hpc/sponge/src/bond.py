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
"""bond class"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class Bond(nn.Cell):
    """bond class"""

    def __init__(self, controller, md_info):
        super(Bond, self).__init__()

        self.atom_numbers = md_info.atom_numbers

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

        self.atom_a = Tensor(np.asarray(self.h_atom_a, np.int32), mstype.int32)
        self.atom_b = Tensor(np.asarray(self.h_atom_b, np.int32), mstype.int32)
        self.bond_k = Tensor(np.asarray(self.h_k, np.float32), mstype.float32)
        self.bond_r0 = Tensor(np.asarray(self.h_r0, np.float32), mstype.float32)

    def process1(self, context):
        """process1: read information from amberfile"""
        for idx, val in enumerate(context):
            if idx < len(context) - 1:
                if "%FLAG POINTERS" in val + context[idx + 1] and "%FORMAT(10I8)" in val + context[idx + 1]:
                    start_idx = idx + 2
                    count = 0
                    value = list(map(int, context[start_idx].strip().split()))
                    self.bond_with_hydrogen = value[2]
                    self.bond_numbers = value[3]
                    self.bond_numbers += self.bond_with_hydrogen
                    print(self.bond_numbers)
                    information = []
                    information.extend(value)
                    while count < 16:
                        start_idx += 1
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                    self.bond_type_numbers = information[15]
                    print("bond type numbers ", self.bond_type_numbers)
                    break

        for idx, val in enumerate(context):
            if "%FLAG BOND_FORCE_CONSTANT" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.bond_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                self.bond_type_k = information[:self.bond_type_numbers]
                break

    def read_information_from_amberfile(self, file_path):
        """read information from amberfile"""
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()
        self.process1(context)

        for idx, val in enumerate(context):
            if "%FLAG BOND_EQUIL_VALUE" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.bond_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                self.bond_type_r = information[:self.bond_type_numbers]
                break

        for idx, val in enumerate(context):
            if "%FLAG BONDS_INC_HYDROGEN" in val:
                self.h_atom_a = [0] * self.bond_numbers
                self.h_atom_b = [0] * self.bond_numbers
                self.h_k = [0] * self.bond_numbers
                self.h_r0 = [0] * self.bond_numbers

                count = 0
                start_idx = idx
                information = []
                while count < 3 * self.bond_with_hydrogen:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)

                for i in range(self.bond_with_hydrogen):
                    self.h_atom_a[i] = information[3 * i + 0] / 3
                    self.h_atom_b[i] = information[3 * i + 1] / 3
                    tmpi = information[3 * i + 2] - 1
                    self.h_k[i] = self.bond_type_k[tmpi]
                    self.h_r0[i] = self.bond_type_r[tmpi]
                break

        for idx, val in enumerate(context):
            if "%FLAG BONDS_WITHOUT_HYDROGEN" in val:
                count = 0
                start_idx = idx
                information = []
                while count < 3 * (self.bond_numbers - self.bond_with_hydrogen):
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)

                for i in range(self.bond_with_hydrogen, self.bond_numbers):
                    self.h_atom_a[i] = information[3 * (i - self.bond_with_hydrogen) + 0] / 3
                    self.h_atom_b[i] = information[3 * (i - self.bond_with_hydrogen) + 1] / 3
                    tmpi = information[3 * (i - self.bond_with_hydrogen) + 2] - 1
                    self.h_k[i] = self.bond_type_k[tmpi]
                    self.h_r0[i] = self.bond_type_r[tmpi]
                break

    def Bond_Energy(self, uint_crd, uint_dr_to_dr_cof):
        """compute bond energy"""
        self.bond_energy = P.BondEnergy(self.bond_numbers, self.atom_numbers)(uint_crd, uint_dr_to_dr_cof, self.atom_a,
                                                                              self.atom_b, self.bond_k, self.bond_r0)
        self.sigma_of_bond_ene = P.ReduceSum()(self.bond_energy)
        return self.sigma_of_bond_ene

    def Bond_Force_With_Atom_Energy(self, uint_crd, scaler):
        """compute bond force with atom energy"""
        self.bfatomenergy = P.BondForceWithAtomEnergy(bond_numbers=self.bond_numbers,
                                                      atom_numbers=self.atom_numbers)
        frc, atom_energy = self.bfatomenergy(uint_crd, scaler, self.atom_a, self.atom_b, self.bond_k, self.bond_r0)
        return frc, atom_energy
