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
"""angle class"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class Angle(nn.Cell):
    """Angle class"""

    def __init__(self, controller):
        super(Angle, self).__init__()
        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

        self.atom_a = Tensor(np.asarray(self.h_atom_a, np.int32), mstype.int32)
        self.atom_b = Tensor(np.asarray(self.h_atom_b, np.int32), mstype.int32)
        self.atom_c = Tensor(np.asarray(self.h_atom_c, np.int32), mstype.int32)
        self.angle_k = Tensor(np.asarray(self.h_angle_k, np.float32), mstype.float32)
        self.angle_theta0 = Tensor(np.asarray(self.h_angle_theta0, np.float32), mstype.float32)

    def read_process1(self, context):
        """read_information_from_amberfile process1"""
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

    def read_process2(self, context):
        """read_information_from_amberfile process2"""
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
        return angle_count

    def read_information_from_amberfile(self, file_path):
        """read information from amberfile"""
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()
        self.read_process1(context)

        self.h_atom_a = [0] * self.angle_numbers
        self.h_atom_b = [0] * self.angle_numbers
        self.h_atom_c = [0] * self.angle_numbers
        self.h_type = [0] * self.angle_numbers
        angle_count = self.read_process2(context)

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
                for i in range(self.angle_without_H_numbers):
                    self.h_atom_a[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 0] / 3
                    self.h_atom_b[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 1] / 3
                    self.h_atom_c[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 2] / 3
                    self.h_type[angle_count] = information[(angle_count - self.angle_with_H_numbers) * 4 + 3] - 1
                    angle_count += 1
                break

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
                        # print(start_idx)
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

    def Angle_Energy(self, uint_crd, uint_dr_to_dr_cof):
        """compute angle energy"""
        self.angle_energy = P.AngleEnergy(self.angle_numbers)(uint_crd, uint_dr_to_dr_cof, self.atom_a, self.atom_b,
                                                              self.atom_c, self.angle_k, self.angle_theta0)
        self.sigma_of_angle_ene = P.ReduceSum()(self.angle_energy)
        return self.sigma_of_angle_ene

    def Angle_Force_With_Atom_Energy(self, uint_crd, scaler):
        """compute angle force with atom energy"""
        print("angele angle numbers:", self.angle_numbers)
        self.afae = P.AngleForceWithAtomEnergy(angle_numbers=self.angle_numbers)
        frc, ene = self.afae(uint_crd, scaler, self.atom_a, self.atom_b, self.atom_c, self.angle_k, self.angle_theta0)
        return frc, ene
