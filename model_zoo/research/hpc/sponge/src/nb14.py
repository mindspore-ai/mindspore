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
'''NON BOND'''
class NON_BOND_14:
    '''NON BOND'''
    def __init__(self, controller, dihedral, atom_numbers):
        self.dihedral_with_hydrogen = dihedral.dihedral_with_hydrogen
        self.dihedral_numbers = dihedral.dihedral_numbers
        self.dihedral_type_numbers = dihedral.dihedral_type_numbers
        self.atom_numbers = atom_numbers

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)
        self.h_atom_a = self.h_atom_a[:self.nb14_numbers]
        self.h_atom_b = self.h_atom_b[:self.nb14_numbers]
        self.h_lj_scale_factor = self.h_lj_scale_factor[:self.nb14_numbers]
        self.h_cf_scale_factor = self.h_cf_scale_factor[:self.nb14_numbers]

    def read_information_from_amberfile(self, file_path):
        '''read amber file'''
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()

        self.cf_scale_type = [0] * self.dihedral_type_numbers
        self.lj_scale_type = [0] * self.dihedral_type_numbers

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
        self.processor(context)

    def processor(self, context):
        '''processor'''
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
