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
"""dihedral class"""

import math
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class Dihedral(nn.Cell):
    """dihedral class"""

    def __init__(self, controller):
        super(Dihedral, self).__init__()
        self.CONSTANT_Pi = 3.1415926535897932
        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)
        self.atom_a = Tensor(np.asarray(self.h_atom_a, np.int32), mstype.int32)
        self.atom_b = Tensor(np.asarray(self.h_atom_b, np.int32), mstype.int32)
        self.atom_c = Tensor(np.asarray(self.h_atom_c, np.int32), mstype.int32)
        self.atom_d = Tensor(np.asarray(self.h_atom_d, np.int32), mstype.int32)

        self.pk = Tensor(np.asarray(self.pk, np.float32), mstype.float32)
        self.gamc = Tensor(np.asarray(self.gamc, np.float32), mstype.float32)
        self.gams = Tensor(np.asarray(self.gams, np.float32), mstype.float32)
        self.pn = Tensor(np.asarray(self.pn, np.float32), mstype.float32)
        self.ipn = Tensor(np.asarray(self.ipn, np.int32), mstype.int32)

    def process1(self, context):
        """process1: read information from amberfile"""
        for idx, val in enumerate(context):
            if idx < len(context) - 1:
                if "%FLAG POINTERS" in val + context[idx + 1] and "%FORMAT(10I8)" in val + context[idx + 1]:
                    start_idx = idx + 2
                    count = 0
                    value = list(map(int, context[start_idx].strip().split()))
                    self.dihedral_with_hydrogen = value[6]
                    self.dihedral_numbers = value[7]
                    self.dihedral_numbers += self.dihedral_with_hydrogen
                    information = []
                    information.extend(value)
                    while count < 15:
                        start_idx += 1
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                    self.dihedral_type_numbers = information[17]
                    print("dihedral type numbers ", self.dihedral_type_numbers)
                    break

        self.phase_type = [0] * self.dihedral_type_numbers
        self.pk_type = [0] * self.dihedral_type_numbers
        self.pn_type = [0] * self.dihedral_type_numbers

        for idx, val in enumerate(context):
            if "%FLAG DIHEDRAL_FORCE_CONSTANT" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.dihedral_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                self.pk_type = information[:self.dihedral_type_numbers]
                break

        for idx, val in enumerate(context):
            if "%FLAG DIHEDRAL_PHASE" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.dihedral_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                self.phase_type = information[:self.dihedral_type_numbers]
                break

        for idx, val in enumerate(context):
            if "%FLAG DIHEDRAL_PERIODICITY" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.dihedral_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                self.pn_type = information[:self.dihedral_type_numbers]
                break

    def read_information_from_amberfile(self, file_path):
        """read information from amberfile"""
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()

        self.process1(context)

        self.h_atom_a = [0] * self.dihedral_numbers
        self.h_atom_b = [0] * self.dihedral_numbers
        self.h_atom_c = [0] * self.dihedral_numbers
        self.h_atom_d = [0] * self.dihedral_numbers
        self.pk = []
        self.gamc = []
        self.gams = []
        self.pn = []
        self.ipn = []
        for idx, val in enumerate(context):
            if "%FLAG DIHEDRALS_INC_HYDROGEN" in val:
                count = 0
                start_idx = idx
                information = []
                while count < 5 * self.dihedral_with_hydrogen:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.dihedral_with_hydrogen):
                    self.h_atom_a[i] = information[i * 5 + 0] / 3
                    self.h_atom_b[i] = information[i * 5 + 1] / 3
                    self.h_atom_c[i] = information[i * 5 + 2] / 3
                    self.h_atom_d[i] = abs(information[i * 5 + 3] / 3)
                    tmpi = information[i * 5 + 4] - 1
                    self.pk.append(self.pk_type[tmpi])
                    tmpf = self.phase_type[tmpi]
                    if abs(tmpf - self.CONSTANT_Pi) <= 0.001:
                        tmpf = self.CONSTANT_Pi
                    tmpf2 = math.cos(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.gamc.append(tmpf2 * self.pk[i])
                    tmpf2 = math.sin(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.gams.append(tmpf2 * self.pk[i])
                    self.pn.append(abs(self.pn_type[tmpi]))
                    self.ipn.append(int(self.pn[i] + 0.001))
                break
        for idx, val in enumerate(context):
            if "%FLAG DIHEDRALS_WITHOUT_HYDROGEN" in val:
                count = 0
                start_idx = idx
                information = []
                while count < 5 * (self.dihedral_numbers - self.dihedral_with_hydrogen):
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.dihedral_with_hydrogen, self.dihedral_numbers):
                    self.h_atom_a[i] = information[(i - self.dihedral_with_hydrogen) * 5 + 0] / 3
                    self.h_atom_b[i] = information[(i - self.dihedral_with_hydrogen) * 5 + 1] / 3
                    self.h_atom_c[i] = information[(i - self.dihedral_with_hydrogen) * 5 + 2] / 3
                    self.h_atom_d[i] = abs(information[(i - self.dihedral_with_hydrogen) * 5 + 3] / 3)
                    tmpi = information[(i - self.dihedral_with_hydrogen) * 5 + 4] - 1
                    self.pk.append(self.pk_type[tmpi])
                    tmpf = self.phase_type[tmpi]
                    if abs(tmpf - self.CONSTANT_Pi) <= 0.001:
                        tmpf = self.CONSTANT_Pi
                    tmpf2 = math.cos(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.gamc.append(tmpf2 * self.pk[i])
                    tmpf2 = math.sin(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.gams.append(tmpf2 * self.pk[i])
                    self.pn.append(abs(self.pn_type[tmpi]))
                    self.ipn.append(int(self.pn[i] + 0.001))
                break
        for i in range(self.dihedral_numbers):
            if self.h_atom_c[i] < 0:
                self.h_atom_c[i] *= -1

    def Dihedral_Engergy(self, uint_crd, uint_dr_to_dr_cof):
        """compute dihedral energy"""
        self.dihedral_energy = P.DihedralEnergy(self.dihedral_numbers)(uint_crd, uint_dr_to_dr_cof, self.atom_a,
                                                                       self.atom_b, self.atom_c, self.atom_d, self.ipn,
                                                                       self.pk, self.gamc, self.gams, self.pn)
        self.sigma_of_dihedral_ene = P.ReduceSum()(self.dihedral_energy)
        return self.sigma_of_dihedral_ene

    def Dihedral_Force_With_Atom_Energy(self, uint_crd, scaler):
        """compute dihedral force and atom energy"""
        self.dfae = P.DihedralForceWithAtomEnergy(dihedral_numbers=self.dihedral_numbers)
        self.frc, self.ene = self.dfae(uint_crd, scaler, self.atom_a, self.atom_b, self.atom_c, self.atom_d,
                                       self.ipn, self.pk, self.gamc, self.gams, self.pn)
        return self.frc, self.ene
