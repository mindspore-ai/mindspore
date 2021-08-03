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
'''virtual_information'''


class VIRTUAL_TYPE_INFROMATION:
    """VIRTUAL_LAYER_INFORMATION"""

    def __init__(self):
        self.virtual_numbers = 0
        self.virtual_type = []


class VIRTUAL_LAYER_INFORMATION:
    '''VIRTUAL_LAYER_INFORMATION'''

    def __init__(self):
        self.v0_info = VIRTUAL_TYPE_INFROMATION()
        self.v1_info = VIRTUAL_TYPE_INFROMATION()
        self.v2_info = VIRTUAL_TYPE_INFROMATION()
        self.v3_info = VIRTUAL_TYPE_INFROMATION()


class Virtual_Information:
    '''virtual_information'''

    def __init__(self, controller, md_info, system_freedom):
        self.module_name = "virtual_atom"
        self.atom_numbers = md_info.atom_numbers
        self.virtual_level = [0] * self.atom_numbers
        self.virtual_layer_info = []
        self.system_freedom = system_freedom
        self.max_level = 1
        self.is_initialized = 0
        name = self.module_name + "_in_file"
        if name in controller.Command_Set:
            self.virtual_layer_info_2 = []
            path = controller.Command_Set[name]
            print("    Start reading virtual levels\n")
            self.read_in_file(path)
            self.max_level, self.total_virtual_atoms = self.level_init()
            self.system_freedom -= 3 * self.total_virtual_atoms
            print("        Virtual Atoms Max Level is ", self.max_level)
            print("        Virtual Atoms Number is ", self.total_virtual_atoms)
            print("    End reading virtual levels")
            self.read_in_file_second(path)
            self.read_in_file_third(path)
            self.is_initialized = 1

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

    def read_in_file(self, path):
        """read_in_file"""
        file = open(path, 'r')
        context = file.readlines()
        line_numbers = 0
        for _, val in enumerate(context):
            line_numbers += 1
            tl = list(map(float, val.strip().split()))
            virtual_type, virtual_atom = int(tl[0]), int(tl[1])
            if virtual_type == 0:
                temp, _ = [int(tl[2])], [tl[3]]
                self.virtual_level[virtual_atom] = self.virtual_level[temp[0]] + 1
            if virtual_type == 1:
                temp, _ = [int(tl[2]), int(tl[3])], [tl[4]]
                self.virtual_level[virtual_atom] = max(self.virtual_level[temp[0]], self.virtual_level[temp[1]]) + 1
            if virtual_type == 2:
                temp, _ = [int(tl[2]), int(tl[3]), int(tl[4])], [tl[5], tl[6]]
                self.virtual_level[virtual_atom] = max(self.virtual_level[temp[0]],
                                                       self.virtual_level[temp[1]],
                                                       self.virtual_level[temp[2]]) + 1
            if virtual_type == 3:
                temp, _ = [int(tl[2]), int(tl[3]), int(tl[4])], [tl[5], tl[6]]
                self.virtual_level[virtual_atom] = max(self.virtual_level[temp[0]],
                                                       self.virtual_level[temp[1]],
                                                       self.virtual_level[temp[2]]) + 1
            if virtual_type > 3 or virtual_type < 0:
                print("        Error: can not parse line #{} because {} is not a proper type for virtual atoms.".format(
                    line_numbers, virtual_type))
                exit(1)
        file.close()

    def level_init(self):
        """level_init"""
        max_level = 0
        total_virtual_atoms = 0
        for i in range(self.atom_numbers):
            vli = self.virtual_level[i]
            if vli > 0:
                total_virtual_atoms += 1
            if vli > max_level:
                for _ in range(vli - max_level):
                    virtual_layer = VIRTUAL_LAYER_INFORMATION()
                    # v0_info.virtual_numbers, v1_info.virtual_numbers
                    # v2_info.virtual_numbers, v3_info.virtual_numbers
                    self.virtual_layer_info.append(virtual_layer)
                max_level = vli
        return max_level, total_virtual_atoms

    def read_in_file_second(self, path):
        """read_in_file_second"""
        print("    Start reading virtual type numbers in different levels")
        file = open(path, 'r')
        context = file.readlines()
        line_numbers = 0
        for _, val in enumerate(context):
            line_numbers += 1
            tl = list(map(float, val.strip().split()))
            virtual_type, virtual_atom = int(tl[0]), int(tl[1])
            temp_vl = self.virtual_layer_info[self.virtual_level[virtual_atom] - 1]
            if virtual_type == 0:
                temp_vl.v0_info.virtual_numbers += 1
            if virtual_type == 1:
                temp_vl.v1_info.virtual_numbers += 1
            if virtual_type == 2:
                temp_vl.v2_info.virtual_numbers += 1
            if virtual_type == 3:
                temp_vl.v3_info.virtual_numbers += 1

            self.virtual_layer_info[self.virtual_level[virtual_atom] - 1] = temp_vl
        print("    End reading virtual type numbers in different levels")

    def read_in_file_third(self, path):
        """read_in_file_third"""
        print("    Start reading information for every virtual atom")
        file = open(path, 'r')
        context = file.readlines()
        line_numbers = 0
        count0, count1, count2, count3 = 0, 0, 0, 0
        temp_v = VIRTUAL_LAYER_INFORMATION()
        self.virtual_layer_info_2 = [0] * len(self.virtual_layer_info)
        for _, val in enumerate(context):
            line_numbers += 1
            tl = list(map(float, val.strip().split()))
            virtual_type, virtual_atom = int(tl[0]), int(tl[1])
            temp_vl = self.virtual_layer_info[self.virtual_level[virtual_atom] - 1]
            if virtual_type == 0:
                temp_vl.v0_info.virtual_type.append([int(tl[1]), int(tl[2]), 2 * tl[3]])
                # virtual_atom, from_1, from_2, from_3, h_double
                count0 += 1
            if virtual_type == 1:
                temp_v.v1_info.virtual_type.append([int(tl[1]), int(tl[2]), int(tl[3]), tl[4]])
                # virtual_atom, from_1, from_2, from_3, a
                count1 += 1
            if virtual_type == 2:
                temp_v.v2_info.virtual_type.append([int(tl[1]), int(tl[2]), int(tl[3]), int(tl[4]), tl[5], tl[6]])
                # virtual_atom, from_1, from_2, from_3, a, b
                count2 += 1
            if virtual_type == 3:
                temp_v.v3_info.virtual_type.append([int(tl[1]), int(tl[2]), int(tl[3]), int(tl[4]), tl[5], tl[6]])
                # virtual_atom, from_1, from_2, from_3, d, k
                count3 += 1
            self.virtual_layer_info_2[self.virtual_level[virtual_atom] - 1] = temp_v
        file.close()
        print("    End reading information for every virtual atom")

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
