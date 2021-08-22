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
'''MD Information'''
import numpy as np
from src.system_information import (periodic_box_condition_information, system_information,
                                    non_bond_information, NVE_iteration, residue_information, trajectory_output)


class md_information:
    '''MD Information'''

    def __init__(self, controller):
        CONSTANT_TIME_CONVERTION = 20.455

        self.md_task = controller.md_task

        self.netfrc = 0 if "net_force" not in controller.Command_Set else int(controller.Command_Set["net_force"])
        self.ntwx = 1000 if "write_information_interval" not in controller.Command_Set else int(
            controller.Command_Set["write_information_interval"])
        self.atom_numbers = 0
        self.residue_numbers = 0
        self.density = 0.0
        self.lin_serial = []
        self.h_res_start = []
        self.h_res_end = []

        self.h_charge = []
        self.h_mass = []
        self.h_mass_inverse = []
        self.h_charge = []
        self.coordinate = []
        self.box_length = []
        self.vel = []
        self.crd = []
        self.velocity = []

        self.mode = self.read_mode(controller)
        # read dt
        self.dt = 0.001 * CONSTANT_TIME_CONVERTION if "dt" not in controller.Command_Set else float(
            controller.Command_Set["dt"]) * CONSTANT_TIME_CONVERTION
        self.dt_in_ps = 0.001 if "dt" not in controller.Command_Set else float(controller.Command_Set["dt"])

        if controller.amber_parm is not None:
            self.read_basic_system_information_from_amber_file(controller.amber_parm)
            if controller.initial_coordinates_file is not None:
                self.read_basic_system_information_from_rst7(controller.initial_coordinates_file)
        else:
            self.read_coordinate_and_velocity(controller)
            self.read_mass(controller)
            self.read_charge(controller)
        self.crd = self.coordinate

        self.sys = system_information(controller, self)
        self.nb = non_bond_information(controller, self)
        self.output = trajectory_output(controller, self)
        self.nve = NVE_iteration(controller, self)
        self.res = residue_information(controller, self)
        self.pbc = periodic_box_condition_information(controller, self.box_length)

        if not self.h_res_start:
            self.h_res_start = self.res.h_res_start
            self.h_res_end = self.res.h_res_end
            self.residue_numbers = self.res.residue_numbers

        # Atom_Information_Initial
        self.acc = np.zeros([self.atom_numbers, 3])
        self.frc = np.zeros([self.atom_numbers, 3])
        self.sys.freedom = 3 * self.atom_numbers
        self.is_initialized = 1

        self.velocity = np.reshape(np.asarray(self.velocity, np.float32), [self.atom_numbers, 3])
        self.step_limit = self.sys.step_limit

    def read_mode(self, controller):
        """read_mode"""
        if "mode" in controller.Command_Set:
            if controller.Command_Set["mode"] in ["NVT", "nvt", "1"]:
                print("    Mode set to NVT\n")
                mode = 1
            elif controller.Command_Set["mode"] in ["NPT", "npt", "2"]:
                print("    Mode set to NPT\n")
                mode = 2
            elif controller.Command_Set["mode"] in ["Minimization", "minimization", "-1"]:
                print("    Mode set to Energy Minimization\n")
                mode = -1
            elif controller.Command_Set["mode"] in ["NVE", "nve", "0"]:
                print("    Mode set to NVE\n")
                mode = 0
            else:
                print(
                    "    Warning: Mode {} is not match. Set to NVE as default\n".format(controller.Command_Set["mode"]))
                mode = 0
        else:
            print("    Mode set to NVE as default\n")
            mode = 0
        return mode

    def read_coordinate_in_file(self, path):
        '''read coordinates file'''
        file = open(path, 'r')
        print("    Start reading coordinate_in_file:\n")
        context = file.readlines()
        atom_numbers = int(context[0].strip())
        if self.atom_numbers != 0:
            if self.atom_numbers is not atom_numbers:
                print("        Error: atom_numbers is not equal: ", atom_numbers, self.atom_numbers)
                exit(1)
        else:
            self.atom_numbers = atom_numbers
            print("        atom_numbers is ", self.atom_numbers)

        for idx in range(self.atom_numbers):
            coord = list(map(float, context[idx + 1].strip().split()))
            self.coordinate.append(coord)

        self.box_length = list(map(float, context[-1].strip().split()))[:3]
        print(" box_length is: x: {}, y: {}, z: {}".format(
            self.box_length[0], self.box_length[1], self.box_length[2]))
        self.crd = self.coordinate
        file.close()

    def read_velocity_in_file(self, path):
        '''read velocity file'''
        file = open(path, 'r')
        print("    Start reading velocity_in_file:\n")
        context = file.readlines()
        for idx, val in enumerate(context):
            if idx == 0:
                atom_numbers = int(val.strip())
                if self.atom_numbers > 0 and atom_numbers != self.atom_numbers:
                    print("        Error: atom_numbers is not equal: %d %d\n", idx, self.atom_numbers)
                    exit(1)
                else:
                    self.atom_numbers = atom_numbers
            else:
                vel = list(map(float, val.strip().split()))
                self.velocity.append(vel)
        self.vel = self.velocity
        file.close()

    def read_coordinate_and_velocity(self, controller):
        """read_coordinate_and_velocity"""
        if "coordinate_in_file" in controller.Command_Set:
            self.read_coordinate_in_file(controller.Command_Set["coordinate_in_file"])
            if "velocity_in_file" in controller.Command_Set:
                self.read_velocity_in_file(controller.Command_Set["velocity_in_file"])
            else:
                print("    Velocity is set to zero as default\n")
                self.velocity = [0] * 3 * self.atom_numbers

    def read_mass(self, controller):
        """read_mass"""
        print("    Start reading mass:")
        if "mass_in_file" in controller.Command_Set:
            path = controller.Command_Set["mass_in_file"]
            file = open(path, 'r')
            self.total_mass = 0
            context = file.readlines()
            for idx, val in enumerate(context):
                if idx == 0:
                    atom_numbers = int(val.strip())
                    if self.atom_numbers > 0 and (atom_numbers != self.atom_numbers):
                        print("        Error: atom_numbers is not equal: ", atom_numbers, self.atom_numbers)
                        exit(1)
                    else:
                        self.atom_numbers = atom_numbers
                else:
                    mass = float(val.strip())
                    self.h_mass.append(mass)
                    self.total_mass += mass
                    if mass == 0:
                        self.h_mass_inverse.append(0.0)
                    else:
                        self.h_mass_inverse.append(1 / mass)
            file.close()
        else:
            print("    mass is set to 20 as default")
            self.total_mass = 20 * self.atom_numbers
            self.h_mass = [20] * self.atom_numbers
            self.h_mass_inverse = [1 / 20] * self.atom_numbers

        print("    End reading mass")

    def read_charge(self, controller):
        """read_charge"""
        if "charge_in_file" in controller.Command_Set:
            print("    Start reading charge:")
            path = controller.Command_Set["charge_in_file"]
            file = open(path, 'r')
            context = file.readlines()
            for idx, val in enumerate(context):
                if idx == 0:
                    atom_numbers = int(val.strip())
                    if self.atom_numbers > 0 and (atom_numbers != self.atom_numbers):
                        print("        Error: atom_numbers is not equal: %d %d\n", idx, self.atom_numbers)
                        exit(1)
                    else:
                        self.atom_numbers = atom_numbers
                else:
                    self.h_charge.append(float(val.strip()))
            file.close()
        else:
            self.h_charge = [0.0] * self.atom_numbers
        print("    End reading charge")

    def read_basic_system_information_from_amber_file(self, path):
        '''read amber file'''
        file = open(path, 'r')
        context = file.readlines()
        for idx, val in enumerate(context):
            if idx < len(context) - 1:
                if "%FLAG POINTERS" in val + context[idx + 1] and "%FORMAT(10I8)" in val + context[idx + 1]:
                    start_idx = idx + 2
                    value = list(map(int, context[start_idx].strip().split()))
                    self.atom_numbers = value[0]
                    count = len(value) - 1
                    while count < 10:
                        start_idx += 1
                        value = list(map(int, context[start_idx].strip().split()))
                        count += len(value)
                    self.residue_numbers = list(map(int, context[start_idx].strip().split()))[
                        10 - (count - 10)]  # may exist bug
                    break

        if self.residue_numbers != 0 and self.atom_numbers != 0:
            for idx, val in enumerate(context):
                if "%FLAG RESIDUE_POINTER" in val:
                    count = 0
                    start_idx = idx
                    while count != self.residue_numbers:
                        start_idx += 1
                        if "%FORMAT" in context[start_idx]:
                            continue
                        else:
                            value = list(map(int, context[start_idx].strip().split()))
                            self.lin_serial.extend(value)
                            count += len(value)
                    for i in range(self.residue_numbers - 1):
                        self.h_res_start.append(self.lin_serial[i] - 1)
                        self.h_res_end.append(self.lin_serial[i + 1] - 1)
                    self.h_res_start.append(self.lin_serial[-1] - 1)
                    self.h_res_end.append(self.atom_numbers + 1 - 1)
                    break
            self.processor(context)

    def processor(self, context):
        '''processor'''
        for idx, val in enumerate(context):
            if "%FLAG MASS" in val:
                count = 0
                start_idx = idx
                while count != self.atom_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        self.h_mass.extend(value)
                        count += len(value)
                for i in range(self.atom_numbers):
                    if self.h_mass[i] == 0:
                        self.h_mass_inverse.append(0.0)
                    else:
                        self.h_mass_inverse.append(1.0 / self.h_mass[i])
                    self.density += self.h_mass[i]
                break
        for idx, val in enumerate(context):
            if "%FLAG CHARGE" in val:
                count = 0
                start_idx = idx
                while count != self.atom_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        self.h_charge.extend(value)
                        count += len(value)
                break

    def read_basic_system_information_from_rst7(self, path):
        '''read rst7 file'''
        file = open(path, 'r')
        context = file.readlines()
        file.close()
        x = context[1].strip().split()
        irest = 1 if len(x) > 1 else 0
        atom_numbers = int(context[1].strip().split()[0])
        if atom_numbers != self.atom_numbers:
            print("ERROR")
        else:
            print("check atom_numbers")
        information = []
        count = 0
        start_idx = 1
        if irest == 1:
            self.simulation_start_time = float(x[1])
            while count <= 6 * self.atom_numbers + 3:
                start_idx += 1
                value = list(map(float, context[start_idx].strip().split()))
                information.extend(value)
                count += len(value)
            self.coordinate = information[: 3 * self.atom_numbers]
            self.velocity = information[3 * self.atom_numbers: 6 * self.atom_numbers]
            self.box_length = information[6 * self.atom_numbers:6 * self.atom_numbers + 3]
        else:
            while count <= 3 * self.atom_numbers + 3:
                start_idx += 1
                value = list(map(float, context[start_idx].strip().split()))
                information.extend(value)
                count += len(value)
            self.coordinate = information[: 3 * self.atom_numbers]
            self.velocity = [0.0] * (3 * self.atom_numbers)
            self.box_length = information[3 * self.atom_numbers:3 * self.atom_numbers + 3]
        self.coordinate = np.array(self.coordinate).reshape([-1, 3])
        self.velocity = np.array(self.velocity).reshape([-1, 3])
        print("system size is ", self.box_length[0], self.box_length[1], self.box_length[2])
