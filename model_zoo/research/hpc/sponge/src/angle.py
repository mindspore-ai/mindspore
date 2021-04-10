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
'''Angle'''
class Angle:
    '''Angle'''
    def __init__(self, controller):
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
                    self.angle_with_H_numbers = value[4]
                    self.angle_without_H_numbers = value[5]
                    self.angle_numbers = self.angle_with_H_numbers + self.angle_without_H_numbers
                    information = []
                    information.extend(value)
                    while count < 15:
                        start_idx += 1
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                    self.angle_type_numbers = information[16]
                    print("angle type numbers ", self.angle_type_numbers)
                    break

        self.h_atom_a = [0] * self.angle_numbers
        self.h_atom_b = [0] * self.angle_numbers
        self.h_atom_c = [0] * self.angle_numbers
        self.h_type = [0] * self.angle_numbers
        angle_count = 0
        for idx, val in enumerate(context):
            if "%FLAG ANGLES_INC_HYDROGEN" in val:
                count = 0
                start_idx = idx
                information = []
                while count < 4 * self.angle_with_H_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for _ in range(self.angle_with_H_numbers):
                    self.h_atom_a[angle_count] = information[angle_count * 4 + 0] / 3
                    self.h_atom_b[angle_count] = information[angle_count * 4 + 1] / 3
                    self.h_atom_c[angle_count] = information[angle_count * 4 + 2] / 3
                    self.h_type[angle_count] = information[angle_count * 4 + 3] - 1
                    angle_count += 1

                break

        for idx, val in enumerate(context):
            if "%FLAG ANGLES_WITHOUT_HYDROGEN" in val:
                count = 0
                start_idx = idx
                information = []
                while count < 4 * self.angle_without_H_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for _ in range(self.angle_without_H_numbers):
                    self.h_atom_a[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 0] / 3
                    self.h_atom_b[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 1] / 3
                    self.h_atom_c[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 2] / 3
                    self.h_type[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 3] - 1
                    angle_count += 1
                break
        self.processor(context, angle_count)

    def processor(self, context, angle_count):
        ''' processor '''
        self.type_k = [0] * self.angle_type_numbers
        for idx, val in enumerate(context):
            if "%FLAG ANGLE_FORCE_CONSTANT" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.angle_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                self.type_k = information[:self.angle_type_numbers]
                break

        self.type_theta0 = [0] * self.angle_type_numbers
        for idx, val in enumerate(context):
            if "%FLAG ANGLE_EQUIL_VALUE" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.angle_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                self.type_theta0 = information[:self.angle_type_numbers]
                break
        if self.angle_numbers != angle_count:
            print("angle count %d != angle_number %d ", angle_count, self.angle_numbers)

        self.h_angle_k = []
        self.h_angle_theta0 = []
        for i in range(self.angle_numbers):
            self.h_angle_k.append(self.type_k[self.h_type[i]])
            self.h_angle_theta0.append(self.type_theta0[self.h_type[i]])
