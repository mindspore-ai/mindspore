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
"""nb14"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class NON_BOND_14(nn.Cell):
    """class Non bond 14"""

    def __init__(self, controller, dihedral, atom_numbers):
        super(NON_BOND_14, self).__init__()

        self.dihedral_with_hydrogen = dihedral.dihedral_with_hydrogen
        self.dihedral_numbers = dihedral.dihedral_numbers
        self.dihedral_type_numbers = dihedral.dihedral_type_numbers
        self.atom_numbers = atom_numbers

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

        self.atom_a = Tensor(np.asarray(self.h_atom_a, np.int32), mstype.int32)
        self.atom_b = Tensor(np.asarray(self.h_atom_b, np.int32), mstype.int32)
        self.lj_scale_factor = Tensor(np.asarray(self.h_lj_scale_factor, np.float32), mstype.float32)
        self.cf_scale_factor = Tensor(np.asarray(self.h_cf_scale_factor, np.float32), mstype.float32)

    def process1(self, context):
        """process1: read information from amberfile"""
        for idx, val in enumerate(context):
            if "%FLAG SCEE_SCALE_FACTOR" in val:
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
                self.cf_scale_type = information[:self.dihedral_type_numbers]
                break

        for idx, val in enumerate(context):
            if "%FLAG SCNB_SCALE_FACTOR" in val:
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
                self.lj_scale_type = information[:self.dihedral_type_numbers]
                break

    def read_information_from_amberfile(self, file_path):
        """read information from amberfile"""
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()

        self.cf_scale_type = [0] * self.dihedral_type_numbers
        self.lj_scale_type = [0] * self.dihedral_type_numbers

        self.h_atom_a = [0] * self.dihedral_numbers
        self.h_atom_b = [0] * self.dihedral_numbers
        self.h_lj_scale_factor = [0] * self.dihedral_numbers
        self.h_cf_scale_factor = [0] * self.dihedral_numbers
        nb14_numbers = 0
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
                    tempa = information[i * 5 + 0]
                    tempi = information[i * 5 + 1]
                    tempi2 = information[i * 5 + 2]
                    tempb = information[i * 5 + 3]
                    tempi = information[i * 5 + 4]

                    tempi -= 1
                    if tempi2 > 0:
                        self.h_atom_a[nb14_numbers] = tempa / 3
                        self.h_atom_b[nb14_numbers] = abs(tempb / 3)
                        self.h_lj_scale_factor[nb14_numbers] = self.lj_scale_type[tempi]
                        if self.h_lj_scale_factor[nb14_numbers] != 0:
                            self.h_lj_scale_factor[nb14_numbers] = 1.0 / self.h_lj_scale_factor[nb14_numbers]
                        self.h_cf_scale_factor[nb14_numbers] = self.cf_scale_type[tempi]
                        if self.h_cf_scale_factor[nb14_numbers] != 0:
                            self.h_cf_scale_factor[nb14_numbers] = 1.0 / self.h_cf_scale_factor[nb14_numbers]
                        nb14_numbers += 1
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
                    tempa = information[(i - self.dihedral_with_hydrogen) * 5 + 0]
                    tempi = information[(i - self.dihedral_with_hydrogen) * 5 + 1]
                    tempi2 = information[(i - self.dihedral_with_hydrogen) * 5 + 2]
                    tempb = information[(i - self.dihedral_with_hydrogen) * 5 + 3]
                    tempi = information[(i - self.dihedral_with_hydrogen) * 5 + 4]

                    tempi -= 1
                    if tempi2 > 0:
                        self.h_atom_a[nb14_numbers] = tempa / 3
                        self.h_atom_b[nb14_numbers] = abs(tempb / 3)
                        self.h_lj_scale_factor[nb14_numbers] = self.lj_scale_type[tempi]
                        if self.h_lj_scale_factor[nb14_numbers] != 0:
                            self.h_lj_scale_factor[nb14_numbers] = 1.0 / self.h_lj_scale_factor[nb14_numbers]
                        self.h_cf_scale_factor[nb14_numbers] = self.cf_scale_type[tempi]
                        if self.h_cf_scale_factor[nb14_numbers] != 0:
                            self.h_cf_scale_factor[nb14_numbers] = 1.0 / self.h_cf_scale_factor[nb14_numbers]
                        nb14_numbers += 1
                break

        self.nb14_numbers = nb14_numbers

    def Non_Bond_14_LJ_Energy(self, uint_crd_with_LJ, uint_dr_to_dr_cof, LJ_A, LJ_B):
        """compute Non bond 14 LJ energy"""
        assert isinstance(uint_crd_with_LJ, tuple)
        uint_crd, LJtype, charge = uint_crd_with_LJ
        self.LJ_energy = P.Dihedral14LJEnergy(self.nb14_numbers, self.atom_numbers)(uint_crd, LJtype, charge,
                                                                                    uint_dr_to_dr_cof, self.atom_a,
                                                                                    self.atom_b, self.lj_scale_factor,
                                                                                    LJ_A, LJ_B)
        self.nb14_lj_energy_sum = P.ReduceSum()(self.LJ_energy)

        return self.nb14_lj_energy_sum

    def Non_Bond_14_CF_Energy(self, uint_crd_with_LJ, uint_dr_to_dr_cof):
        """compute Non bond 14 CF energy"""
        assert isinstance(uint_crd_with_LJ, tuple)
        uint_crd, LJtype, charge = uint_crd_with_LJ
        self.CF_energy = P.Dihedral14CFEnergy(self.nb14_numbers, self.atom_numbers)(uint_crd, LJtype, charge,
                                                                                    uint_dr_to_dr_cof, self.atom_a,
                                                                                    self.atom_b, self.cf_scale_factor)
        self.nb14_cf_energy_sum = P.ReduceSum()(self.CF_energy)
        return self.nb14_cf_energy_sum

    def Non_Bond_14_LJ_CF_Energy(self, uint_crd_with_LJ, uint_dr_to_dr_cof, LJ_A, LJ_B):
        """compute Non bond 14 LJ and CF energy"""
        assert isinstance(uint_crd_with_LJ, tuple)
        self.nb14_lj_energy_sum = self.Non_Bond_14_LJ_Energy(uint_crd_with_LJ, uint_dr_to_dr_cof, LJ_A, LJ_B)
        self.nb14_cf_energy_sum = self.Non_Bond_14_CF_Energy(uint_crd_with_LJ, uint_dr_to_dr_cof)

        return self.nb14_lj_energy_sum, self.nb14_cf_energy_sum

    def Non_Bond_14_LJ_CF_Force_With_Atom_Energy(self, uint_crd_with_LJ, boxlength, LJ_A, LJ_B):
        """compute Non bond 14 LJ CF force and atom energy"""
        self.d14lj = P.Dihedral14LJCFForceWithAtomEnergy(nb14_numbers=self.nb14_numbers, atom_numbers=self.atom_numbers)
        assert isinstance(uint_crd_with_LJ, tuple)
        uint_crd_f, LJtype, charge = uint_crd_with_LJ
        self.frc, self.atom_ene = self.d14lj(uint_crd_f, LJtype, charge, boxlength, self.atom_a, self.atom_b,
                                             self.lj_scale_factor, self.cf_scale_factor, LJ_A, LJ_B)
        return self.frc, self.atom_ene
