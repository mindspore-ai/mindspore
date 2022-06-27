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
'''Dihedral'''
import math


class Dihedral:
    '''Dihedral'''

    def __init__(self, controller):
        self.constant_pi = 3.1415926535897932
        self.module_name = "dihedral"
        self.h_atom_a = []
        self.h_atom_b = []
        self.h_atom_c = []
        self.h_atom_d = []
        self.h_ipn = []
        self.h_pn = []
        self.h_pk = []
        self.h_gamc = []
        self.h_gams = []
        self.dihedral_numbers = 0
        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)
            self.is_initialized = 1
        else:
            self.read_in_file(controller)

    def read_in_file(self, controller):
        """read_in_file"""
        print("START INITIALIZING DIHEDRAL:")
        name = self.module_name + "_in_file"
        if name in controller.command_set:
            path = controller.command_set[name]
            file = open(path, 'r')
            context = file.readlines()
            self.dihedral_numbers = int(context[0].strip())
            print("    dihedral_numbers is ", self.dihedral_numbers)
            for i in range(self.dihedral_numbers):
                val = list(map(float, context[i + 1].strip().split()))
                self.h_atom_a.append(int(val[0]))
                self.h_atom_b.append(int(val[1]))
                self.h_atom_c.append(int(val[2]))
                self.h_atom_d.append(int(val[3]))
                self.h_ipn.append(val[4])
                self.h_pn.append(val[4])
                self.h_pk.append(val[5])
                self.h_gamc.append(math.cos(val[6]) * val[5])
                self.h_gams.append(math.sin(val[6]) * val[5])

            self.is_initialized = 1
            file.close()
        print("END INITIALIZING DIHEDRAL")

    def read_information_from_amberfile(self, file_path):
        '''read amber file'''
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()
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
        self.processor(context)

    def processor(self, context):
        '''processor'''
        self.h_atom_a = [0] * self.dihedral_numbers
        self.h_atom_b = [0] * self.dihedral_numbers
        self.h_atom_c = [0] * self.dihedral_numbers
        self.h_atom_d = [0] * self.dihedral_numbers
        self.h_pk = []
        self.h_gamc = []
        self.h_gams = []
        self.h_pn = []
        self.h_ipn = []
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
                    self.h_pk.append(self.pk_type[tmpi])
                    tmpf = self.phase_type[tmpi]
                    if abs(tmpf - self.constant_pi) <= 0.001:
                        tmpf = self.constant_pi
                    tmpf2 = math.cos(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.h_gamc.append(tmpf2 * self.h_pk[i])
                    tmpf2 = math.sin(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.h_gams.append(tmpf2 * self.h_pk[i])
                    self.h_pn.append(abs(self.pn_type[tmpi]))
                    self.h_ipn.append(int(self.h_pn[i] + 0.001))
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
                    self.h_pk.append(self.pk_type[tmpi])
                    tmpf = self.phase_type[tmpi]
                    if abs(tmpf - self.constant_pi) <= 0.001:
                        tmpf = self.constant_pi
                    tmpf2 = math.cos(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.h_gamc.append(tmpf2 * self.h_pk[i])
                    tmpf2 = math.sin(tmpf)
                    if abs(tmpf2) < 1e-6:
                        tmpf2 = 0
                    self.h_gams.append(tmpf2 * self.h_pk[i])
                    self.h_pn.append(abs(self.pn_type[tmpi]))
                    self.h_ipn.append(int(self.h_pn[i] + 0.001))
                break
        for i in range(self.dihedral_numbers):
            if self.h_atom_c[i] < 0:
                self.h_atom_c[i] *= -1
