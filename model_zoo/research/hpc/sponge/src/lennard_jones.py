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
'''Lennard Jones'''
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P


class Lennard_Jones_Information:
    '''Lennard Jones'''

    def __init__(self, controller, cutoff, box_length):
        self.module_name = "LJ"
        self.is_initialized = 0
        self.CONSTANT_UINT_MAX_FLOAT = 4294967296.0
        self.CONSTANT_Pi = 3.1415926535897932
        self.cutoff = cutoff
        self.box_length = box_length

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)
            self.is_initialized = 1
        else:
            self.read_in_file(controller)

        if self.is_initialized:
            self.totalc6get = P.totalc6get(self.atom_numbers)
            self.read_information()

    def read_in_file(self, controller):
        """read_in_file"""
        print("START INITIALIZING LENNADR JONES INFORMATION:")
        name = self.module_name + "_in_file"
        # print("read_in_file " + name)
        if name in controller.Command_Set:
            path = controller.Command_Set[name]
            file = open(path, 'r')
            context = file.readlines()
            self.atom_numbers, self.atom_type_numbers = map(int, context[0].strip().split())
            print("    atom_numbers is ", self.atom_numbers)
            print("    atom_LJ_type_number is ", self.atom_type_numbers)
            self.pair_type_numbers = self.atom_type_numbers * (self.atom_type_numbers + 1) / 2
            self.h_LJ_A = []
            self.h_LJ_B = []
            self.h_atom_LJ_type = []
            startidx = 1
            count = 0
            print(startidx)
            while count < self.atom_type_numbers:
                if context[startidx].strip():
                    val = list(map(float, context[startidx].strip().split()))
                    # print(val)
                    count += 1
                    self.h_LJ_A.extend(val)
                startidx += 1
            assert len(self.h_LJ_A) == self.pair_type_numbers
            self.h_LJ_A = [x * 12.0 for x in self.h_LJ_A]

            count = 0
            print(startidx)
            while count < self.atom_type_numbers:
                if context[startidx].strip():
                    val = list(map(float, context[startidx].strip().split()))
                    # print(val)
                    count += 1
                    self.h_LJ_B.extend(val)
                startidx += 1
            assert len(self.h_LJ_B) == self.pair_type_numbers
            self.h_LJ_B = [x * 6.0 for x in self.h_LJ_B]
            for idx, val in enumerate(context):
                if idx > startidx:
                    self.h_atom_LJ_type.append(int(val.strip()))
            file.close()
            self.is_initialized = 1
        print("END INITIALIZING LENNADR JONES INFORMATION")

    def read_information(self):
        """read_information"""
        self.uint_dr_to_dr_cof = [1.0 / self.CONSTANT_UINT_MAX_FLOAT * self.box_length[0],
                                  1.0 / self.CONSTANT_UINT_MAX_FLOAT * self.box_length[1],
                                  1.0 / self.CONSTANT_UINT_MAX_FLOAT * self.box_length[2]]
        print("copy lj type to new crd")
        self.atom_LJ_type = Tensor(self.h_atom_LJ_type, mstype.int32)
        self.LJ_B = Tensor(self.h_LJ_B, mstype.float32)
        self.factor = self.totalc6get(self.atom_LJ_type, self.LJ_B)
        print("        factor is: ", self.factor)
        self.long_range_factor = float(self.factor.asnumpy())
        self.long_range_factor *= -2.0 / 3.0 * self.CONSTANT_Pi / self.cutoff / self.cutoff / self.cutoff / 6.0
        self.volume = self.box_length[0] * self.box_length[1] * self.box_length[1]
        print("        long range correction factor is: ", self.long_range_factor)
        print("    End initializing long range LJ correction")

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
                    self.atom_numbers = value[0]
                    self.atom_type_numbers = value[1]
                    self.pair_type_numbers = int(
                        self.atom_type_numbers * (self.atom_type_numbers + 1) / 2)  # TODO
                    break
        self.h_atom_LJ_type = [0] * self.atom_numbers
        for idx, val in enumerate(context):
            if "%FLAG ATOM_TYPE_INDEX" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.atom_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.atom_numbers):
                    self.h_atom_LJ_type[i] = information[i] - 1
                break
        self.h_LJ_A = [0] * self.pair_type_numbers
        for idx, val in enumerate(context):
            if "%FLAG LENNARD_JONES_ACOEF" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.pair_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.pair_type_numbers):
                    self.h_LJ_A[i] = 12.0 * information[i]
                break
        self.h_LJ_B = [0] * self.pair_type_numbers
        for idx, val in enumerate(context):
            if "%FLAG LENNARD_JONES_BCOEF" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.pair_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.pair_type_numbers):
                    self.h_LJ_B[i] = 6.0 * information[i]
                break
