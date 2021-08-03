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
'''System information'''
import numpy as np


class periodic_box_condition_information:
    """periodic_box_condition_information"""

    def __init__(self, controller, box_length):
        CONSTANT_UINT_MAX_FLOAT = 4294967296.0
        self.crd_to_uint_crd_cof = np.array([CONSTANT_UINT_MAX_FLOAT / box_length[0],
                                             CONSTANT_UINT_MAX_FLOAT / box_length[1],
                                             CONSTANT_UINT_MAX_FLOAT / box_length[2]])
        self.quarter_crd_to_uint_crd_cof = 0.25 * self.crd_to_uint_crd_cof
        self.uint_dr_to_dr_cof = 1.0 / self.crd_to_uint_crd_cof


class system_information:
    """system_information"""

    def __init__(self, controller, md_info):
        CONSTANT_PRES_CONVERTION_INVERSE = 0.00001439506089041446
        self.md_info = md_info
        self.box_length = self.md_info.box_length
        self.steps = 0
        self.step_limit = 1000 if "step_limit" not in controller.Command_Set else int(
            controller.Command_Set["step_limit"])
        self.target_temperature = 300.0 if "target_temperature" not in controller.Command_Set else float(
            controller.Command_Set["target_temperature"])
        if md_info.mode == 2 and "target_pressure" in controller.Command_Set:
            self.target_pressure = float(controller.Command_Set["target_pressure"])
        else:
            self.target_pressure = 1
        self.target_pressure *= CONSTANT_PRES_CONVERTION_INVERSE
        self.d_virial = 0
        self.d_pressure = 0
        self.d_temperature = 0
        self.d_potential = 0
        self.d_sum_of_atom_ek = 0
        self.freedom = 3 * md_info.atom_numbers


class non_bond_information:
    """system_information"""

    def __init__(self, controller, md_info):
        self.md_info = md_info
        self.skin = 2.0 if "skin" not in controller.Command_Set else float(controller.Command_Set["skin"])
        print("    skin set to %.2f Angstram" % (self.skin))
        self.cutoff = 10.0 if "cutoff" not in controller.Command_Set else float(controller.Command_Set["cutoff"])
        self.atom_numbers = self.md_info.atom_numbers
        self.excluded_atom_numbers = 0
        self.h_excluded_list_start = []
        self.h_excluded_numbers = []
        self.h_excluded_list = []
        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)
        else:
            self.read_exclude_file(controller)

    def read_exclude_file(self, controller):
        """read_exclude_file"""
        if "exclude_in_file" in controller.Command_Set:
            print("    Start reading excluded list:")
            path = controller.Command_Set["exclude_in_file"]
            file = open(path, 'r')
            context = file.readlines()
            atom_numbers, self.excluded_atom_numbers = list(map(int, context[0].strip().split()))
            if self.md_info.atom_numbers > 0 and (atom_numbers != self.md_info.atom_numbers):
                print("        Error: atom_numbers is not equal: ", atom_numbers, self.md_info.atom_numbers)
                exit(1)
            else:
                self.md_info.atom_numbers = atom_numbers
            count = 0
            for idx, val in enumerate(context):
                if idx > 0:
                    el = list(map(int, val.strip().split()))
                    if el[0] == 1 and -1 in el:
                        self.h_excluded_numbers.append(0)
                    else:
                        self.h_excluded_numbers.append(el[0])
                    self.h_excluded_list_start.append(count)
                    if el:
                        self.h_excluded_list.extend(el[1:])
                        count += el[0]
            print("    End reading excluded list")
            file.close()
        else:
            print("    Set all atom exclude no atoms as default")
            count = 0
            for i in range(self.md_info.atom_numbers):
                self.h_excluded_numbers[i] = 0
                self.h_excluded_list_start[i] = count
                for _ in range(self.h_excluded_numbers[i]):
                    self.h_excluded_list[count] = 0
                    count += 1
            print("    End reading charge")

    def read_information_from_amberfile(self, file_path):
        '''read amber file'''
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()
        self.h_excluded_list_start = [0] * self.atom_numbers
        self.h_excluded_numbers = [0] * self.atom_numbers

        for idx, val in enumerate(context):
            if idx < len(context) - 1:
                if "%FLAG POINTERS" in val + context[idx + 1] and "%FORMAT(10I8)" in val + context[idx + 1]:
                    start_idx = idx + 2
                    count = 0
                    value = list(map(int, context[start_idx].strip().split()))
                    information = []
                    information.extend(value)
                    while count < 11:
                        start_idx += 1
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                    self.excluded_atom_numbers = information[10]
                    print("excluded atom numbers ", self.excluded_atom_numbers)
                    break
        for idx, val in enumerate(context):
            if "%FLAG NUMBER_EXCLUDED_ATOMS" in val:
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
                count = 0
                for i in range(self.atom_numbers):
                    self.h_excluded_numbers[i] = information[i]
                    self.h_excluded_list_start[i] = count
                    count += information[i]
                break

        total_count = sum(self.h_excluded_numbers)
        self.h_excluded_list = []
        for idx, val in enumerate(context):
            if "%FLAG EXCLUDED_ATOMS_LIST" in val:
                count = 0
                start_idx = idx
                information = []
                while count < total_count:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)

                count = 0
                for i in range(self.atom_numbers):
                    tmp_list = []
                    if self.h_excluded_numbers[i] == 1:
                        tmp_list.append(information[count] - 1)
                        if information[count] == 0:
                            self.h_excluded_numbers[i] = 0
                        count += 1
                    else:
                        for _ in range(self.h_excluded_numbers[i]):
                            tmp_list.append(information[count] - 1)

                            count += 1
                        tmp_list = sorted(tmp_list)
                    self.h_excluded_list.extend(tmp_list)
                break


class NVE_iteration:
    """NVE_iteration"""

    def __init__(self, controller, md_info):
        self.max_velocity = -1 if "nve_velocity_max" not in controller.Command_Set else float(
            controller.Command_Set["nve_velocity_max"])


class residue_information:
    """residue_information"""

    def __init__(self, controller, md_info):
        self.md_info = md_info
        self.residue_numbers = 0
        self.h_mass = []
        self.h_mass_inverse = []
        self.h_res_start = []
        self.h_res_end = []
        self.momentum = []
        self.center_of_mass = []
        self.sigma_of_res_ek = 0
        self.res_ek_energy = 0
        self.sigma_of_res_ek = 0
        self.is_initialized = 0
        print("    Start reading residue list:")
        if "residue_in_file" in controller.Command_Set:
            self.read_residule_file(controller)
        elif "amber_parm7" in controller.Command_Set:
            self.residue_numbers = self.md_info.residue_numbers
            self.h_res_start = md_info.h_res_start
            self.h_res_end = md_info.h_res_end
            self.is_initialized = 1
            self.read_res_mass()
        else:
            self.residue_numbers = md_info.atom_numbers
            self.h_res_start = list(range(self.residue_numbers))
            self.h_res_end = list(range(1, self.residue_numbers + 1))
            self.is_initialized = 1
        print("    End reading residue list")

    def read_res_mass(self):
        """ Read_AMBER_Parm7 """
        if self.md_info.h_mass:
            for i in range(self.residue_numbers):
                temp_mass = 0
                for j in range(self.h_res_start[i], self.h_res_end[i]):
                    temp_mass += self.md_info.h_mass[j]
                self.h_mass.append(temp_mass)
                if temp_mass == 0:
                    self.h_mass_inverse.append(0)
                else:
                    self.h_mass_inverse.append(1.0 / temp_mass)
        else:
            print("    Error: atom mass should be initialized before residue mass")
            exit(1)

    def read_residule_file(self, controller):
        """read_residule_file"""
        if "residue_in_file" in controller.Command_Set:
            path = controller.Command_Set["residue_in_file"]
            file = open(path, 'r')
            context = file.readlines()
            atom_numbers, self.residue_numbers = list(map(int, context[0].strip().split()))
            print("        residue_numbers is ", self.residue_numbers)
            # self.md_info.residue_numbers = self.residue_numbers
            if self.md_info.atom_numbers > 0 and (atom_numbers != self.md_info.atom_numbers):
                print("        Error: atom_numbers is not equal: ", atom_numbers, self.md_info.atom_numbers)
                exit(1)
            else:
                self.md_info.atom_numbers = atom_numbers
            print("        residue_numbers is ", self.residue_numbers)

            count = 0
            for idx, val in enumerate(context):
                if idx > 0:
                    self.h_res_start.append(count)
                    temp = int(val.strip())
                    count += temp
                    self.h_res_end.append(count)
            print("    End reading excluded list")
            file.close()
            self.is_initialized = 1
            if self.is_initialized:
                self.read_res_mass()


class trajectory_output:
    """trajectory_output"""

    def __init__(self, controller, md_info):
        self.current_crd_synchronized_step = 0
        self.is_molecule_map_output = 0
        if "molecule_map_output" in controller.Command_Set:
            self.is_molecule_map_output = int(controller.Command_Set["molecule_map_output"])
        self.amber_irest = -1
        self.write_trajectory_interval = 1000 if "write_information_interval" not in controller.Command_Set else int(
            controller.Command_Set["write_information_interval"])
        self.write_restart_file_interval = self.write_trajectory_interval if "write_restart_file_interval" not in \
                                                                             controller.Command_Set else \
            int(controller.Command_Set["write_restart_file_interval"])
