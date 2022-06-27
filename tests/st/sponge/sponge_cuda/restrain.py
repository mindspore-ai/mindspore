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
'''restrain_information'''


class RestrainInformation:
    '''restrain_information'''

    def __init__(self, controller, atom_numbers, crd):
        self.module_name = "restrain"
        self.atom_numbers = atom_numbers
        self.crd = crd
        self.controller = controller
        self.weight = 100.0 if "weight" not in controller.command_set else float(
            controller.command_set["weight"])
        print("    %s_weight is %.0f\n" % (self.module_name, self.weight))
        self.is_initialized = 0
        name = self.module_name + "_in_file"
        if name in controller.command_set:
            print("START INITIALIZING RESTRAIN\n")
            path = controller.command_set[name]
            self.read_in_file(path)
            self.read_crd_ref()
            self.is_initialized = 1
            print("END INITIALIZING RESTRAIN\n")

    def read_in_file(self, path):
        """read_in_file"""
        file = open(path, 'r')
        context = file.readlines()
        self.restrain_numbers = 0
        h_lists = []
        for _, val in enumerate(context):
            h_lists.append(float(val.strip()))
            self.restrain_numbers += 1
        print("        restrain_numbers is %d\n", self.restrain_numbers)
        file.close()
        self.restrain_ene = [0] * self.restrain_numbers
        self.sum_of_restrain_ene = [0]

    def read_crd_ref(self):
        """read_crd_ref"""
        self.crd_ref = []
        if "coordinate" not in self.controller.command_set:
            print("    restrain reference coordinate copy from input coordinate")
            self.crd_ref = self.crd
        else:
            print("    reading restrain reference coordinate file")
            file = open(self.controller.command_set["coordinate"], 'r')
            context = file.readlines()
            atom_numbers = int(context[0].strip())
            print("        atom_numbers is %d", atom_numbers)
            for i in range(atom_numbers):
                self.crd_ref.append(list(map(float, context[i + 1].strip().split())))
