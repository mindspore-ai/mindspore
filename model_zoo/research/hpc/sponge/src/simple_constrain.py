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
'''Simple_Constarin'''''
import math


class Bond_Information:
    def __init__(self):
        self.bond_numbers = 0
        self.atom_a = []
        self.atom_b = []
        self.bond_r = []


class Information:
    '''information'''

    def __init__(self):
        self.atom_numbers = 0
        self.dt = 0.001
        self.uint_dr_to_dr_cof = []
        self.quarter_crd_to_uint_crd_cof = []
        self.volume = 0.0
        self.exp_gamma = 1.0
        self.step_length = 1.0
        self.iteration_numbers = 25
        self.constrain_mass = 3


class Simple_Constarin:
    '''Simple_Constarin'''

    def __init__(self, controller, md_info, bond, angle, liujian_info):
        self.module_name = "simple_constrain"
        self.CONSTANT_UINT_MAX_FLOAT = 4294967296.0
        self.controller = controller
        self.md_info = md_info
        self.bond = bond
        self.angle = angle
        self.liujian_info = liujian_info
        self.bond_info = Bond_Information()
        self.info = Information()

        self.atom_numbers = md_info.atom_numbers
        self.bond_numbers = bond.bond_numbers
        self.angle_numbers = angle.angle_numbers

        self.is_initialized = 0
        self.constrain_frc = []
        self.pair_virial = []
        self.virial = []
        self.test_uint_crd = []
        self.last_pair_dr = []
        self.dt_inverse = 0
        self.half_exp_gamma_plus_half = 0
        self.bond_constrain_pair_numbers = 0
        self.angle_constrain_pair_numbers = 0
        self.constrain_pair_numbers = 0
        self.h_bond_pair = []
        self.h_constrain_pair = []
        if "constrain_mode" in self.controller.Command_Set and \
                self.controller.Command_Set["constrain_mode"] == "simple_constrain":
            print("START INITIALIZING SIMPLE CONSTRAIN:")
            self.constrain_pair_numbers = 0
            self.add_hbond_to_constrain_pair(self.bond_numbers, bond.h_atom_a, bond.h_atom_b, bond.h_r0, md_info.h_mass)
            self.add_hangle_to_constrain_pair(self.angle_numbers, angle.h_atom_a, angle.h_atom_b,
                                              angle.h_atom_c, angle.h_angle_theta0, md_info.h_mass)
            if liujian_info.is_initialized:
                self.initial_simple_constrain(md_info.atom_numbers, md_info.dt, md_info.sys.box_length,
                                              liujian_info.exp_gamma, 0, md_info.h_mass, md_info.sys.freedom)
            else:
                self.initial_simple_constrain(md_info.atom_numbers, md_info.dt, md_info.sys.box_length,
                                              1.0, md_info.mode == -1, md_info.h_mass, md_info.sys.freedom)
            self.is_initialized = 1
            print("END INITIALIZING SIMPLE CONSTRAIN\n")

    def add_hbond_to_constrain_pair(self, bond_numbers, atom_a, atom_b, bond_r, atom_mass):
        """add_hbond_to_constrain_pair"""
        self.info.constrain_mass = 3.0
        name = self.module_name + "_in_file"
        if name in self.controller.Command_Set:
            self.info.constrain_mass = 0
        if "mass" in self.controller.Command_Set:
            self.info.constrain_mass = float(self.controller.Command_Set["mass"])
        self.h_bond_pair = []
        s = 0
        for i in range(bond_numbers):
            mass_a = atom_mass[atom_a[i]]
            mass_b = atom_mass[atom_b[i]]
            if (0 < mass_a < self.info.constrain_mass) or (0 < mass_b < self.info.constrain_mass):
                constrain_k = atom_mass[atom_a[i]] * atom_mass[atom_b[i]] / \
                              (atom_mass[atom_a[i]] + atom_mass[atom_b[i]])
                self.h_bond_pair.append([atom_a[i], atom_b[i], bond_r[i], constrain_k])
                s += 1
        self.bond_constrain_pair_numbers = s
        self.bond_info.bond_numbers = bond_numbers
        self.bond_info.atom_a = atom_a
        self.bond_info.atom_b = atom_b
        self.bond_info.bond_r = bond_r

    def add_hangle_to_constrain_pair(self, angle_numbers, atom_a, atom_b, atom_c, angle_theta, atom_mass):
        """add_hbond_to_constrain_pair"""
        self.h_angle_pair = []
        s = 0
        for i in range(angle_numbers):
            mass_a = atom_mass[atom_a[i]]
            mass_c = atom_mass[atom_c[i]]
            if ((0 < mass_a < self.info.constrain_mass) or (0 < mass_c < self.info.constrain_mass)):
                for j in range(self.bond_info.bond_numbers):
                    if ((self.bond_info.atom_a[j] == atom_a[i] and self.bond_info.atom_b[j] == atom_b[i])
                            or (self.bond_info.atom_a[j] == atom_b[i] and self.bond_info.atom_b[j] == atom_a[i])):
                        rab = self.bond_info.bond_r[j]
                    if ((self.bond_info.atom_a[j] == atom_c[i] and self.bond_info.atom_b[j] == atom_b[i])
                            or (self.bond_info.atom_a[j] == atom_b[i] and self.bond_info.atom_b[j] == atom_c[i])):
                        rbc = self.bond_info.bond_r[j]

                constant_r = math.sqrt(rab * rab + rbc * rbc - 2. * rab * rbc * math.cos(angle_theta[i]))
                constrain_k = atom_mass[atom_a[i]] * atom_mass[atom_c[i]] / \
                              (atom_mass[atom_a[i]] + atom_mass[atom_c[i]])
                self.h_angle_pair.append([atom_a[i], atom_c[i], constant_r, constrain_k])
                s = s + 1
        self.angle_constrain_pair_numbers = s

    def initial_simple_constrain(self, atom_numbers, dt, box_length, exp_gamma, is_Minimization, atom_mass,
                                 system_freedom):
        """initial_simple_constrain"""
        self.system_freedom = system_freedom
        self.atom_mass = atom_mass
        self.info.atom_numbers = atom_numbers
        self.info.dt = dt
        self.info.quarter_crd_to_uint_crd_cof = [0.25 * self.CONSTANT_UINT_MAX_FLOAT / box_length[0],
                                                 0.25 * self.CONSTANT_UINT_MAX_FLOAT / box_length[1],
                                                 0.25 * self.CONSTANT_UINT_MAX_FLOAT / box_length[2]]
        self.info.uint_dr_to_dr_cof = [1 / self.CONSTANT_UINT_MAX_FLOAT * box_length[0],
                                       1 / self.CONSTANT_UINT_MAX_FLOAT * box_length[1],
                                       1 / self.CONSTANT_UINT_MAX_FLOAT * box_length[2]]
        self.info.volume = box_length[0] * box_length[1] * box_length[2]
        self.info.exp_gamma = exp_gamma
        self.half_exp_gamma_plus_half = 0.5 * (1. + self.info.exp_gamma)
        if is_Minimization:
            self.info.exp_gamma = 0.0
        self.info.iteration_numbers = 25 if "iteration_numbers" not in self.controller.Command_Set else int(
            self.controller.Command_Set["iteration_numbers"])
        print("    constrain iteration step is ", self.info.iteration_numbers)
        self.info.step_length = 1 if "step_length" not in self.controller.Command_Set else float(
            self.controller.Command_Set["step_length"])
        print("    constrain step_length is ", self.info.step_length)
        self.extra_numbers = 0
        name = self.module_name + "_in_file"
        if name in self.controller.Command_Set:
            path = self.controller.Command_Set[name]
            file = open(path, 'r')
            context = file.readlines()
            self.extra_numbers = int(context[0].strip())
            file.close()

        self.dt_inverse = 1 / self.info.dt
        self.constrain_pair_numbers = self.bond_constrain_pair_numbers + self.angle_constrain_pair_numbers + \
                                      self.extra_numbers
        self.system_freedom -= self.constrain_pair_numbers

        print("    constrain pair number is ", self.constrain_pair_numbers)

        for i in range(self.bond_constrain_pair_numbers):
            self.h_constrain_pair.append(self.h_bond_pair[i])
            self.h_constrain_pair[i][-1] = self.info.step_length / self.half_exp_gamma_plus_half \
                                           * self.h_constrain_pair[i][-1]
        for i in range(self.angle_constrain_pair_numbers):
            self.h_constrain_pair.append(self.h_bond_pair[i])
            idx = self.bond_constrain_pair_numbers
            self.h_constrain_pair[i + idx][-1] = self.info.step_length / \
                                                 self.half_exp_gamma_plus_half * self.h_constrain_pair[i + idx][-1]
        if name in self.controller.Command_Set:
            path = self.controller.Command_Set[name]
            self.read_in_file(path)

        self.is_initialized = 1

    def read_in_file(self, path):
        """read_in_file"""
        file = open(path, 'r')
        context = file.readlines()
        count = self.bond_constrain_pair_numbers + self.angle_constrain_pair_numbers
        for i in range(self.extra_numbers):
            val = list(map(float, context[i + 1].strip().split()))
            atom_i, atom_j, constant_r = int(val[0]), int(val[1]), float(val[2])
            constrain_k = self.info.step_length / self.half_exp_gamma_plus_half * self.atom_mass[atom_i] * \
                          self.atom_mass[atom_j] / (self.atom_mass[atom_i] + self.atom_mass[atom_j])
            self.h_constrain_pair.append([atom_i, atom_j, constant_r, constrain_k])
            count += 1
        file.close()
