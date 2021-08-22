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
'''Bond'''
class Bond:
    '''Bond'''
    def __init__(self, controller, md_info):

        self.atom_numbers = md_info.atom_numbers

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

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
                    self.bond_with_hydrogen = value[2]
                    self.bond_numbers = value[3]
                    self.bond_numbers += self.bond_with_hydrogen
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
        self.processor(context)

    def processor(self, context):
        '''processor'''
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
