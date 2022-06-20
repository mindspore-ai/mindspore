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
"""SimpleConstarin"""


class BondInformation:
    def __init__(self):
        self.bond_numbers = 0
        self.atom_a = []
        self.atom_b = []
        self.bond_r = []


class Information:
    """Information"""

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


class SimpleConstarin:
    """SimpleConstrain"""

    def __init__(self, controller, md_info, bond, angle, liujian_info):
        self.module_name = "simple_constrain"
        self.constant_unit_max_float = 4294967296.0
        self.controller = controller
        self.md_info = md_info
        self.bond = bond
        self.angle = angle
        self.liujian_info = liujian_info
        self.bond_info = BondInformation()
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
        # in the real calculation, use constrain pair
        self.constrain_pair_numbers = 1
        self.system_freedom = 0
        self.h_bond_pair = []
        self.h_constrain_pair = []
        if "constrain_mode" in self.controller.command_set and \
                self.controller.command_set["constrain_mode"] == "simple_constrain":
            print("START INITIALIZING SIMPLE CONSTRAIN:")
            self.constrain_pair_numbers = 0
            self.add_hbond_to_constrain_pair(self.bond_numbers, bond.h_atom_a, bond.h_atom_b, bond.h_r0, md_info.h_mass)
            self.add_hangle_to_constrain_pair()
            if liujian_info.is_initialized:
                self.initial_simple_constrain(md_info.atom_numbers, md_info.dt, md_info.sys.box_length,
                                              liujian_info.exp_gamma, 0, md_info.h_mass, md_info.sys.freedom)
            else:
                self.initial_simple_constrain(md_info.atom_numbers, md_info.dt, md_info.sys.box_length,
                                              1.0, md_info.mode == -1, md_info.h_mass, md_info.sys.freedom)
            self.is_initialized = 1
            print("END INITIALIZING SIMPLE CONSTRAIN\n")

    def add_hbond_to_constrain_pair(self, bond_numbers, atom_a, atom_b, bond_r, atom_mass):
        """ add hbond to constrain pair
        :param bond_numbers:
        :param atom_a:
        :param atom_b:
        :param bond_r:
        :param atom_mass:
        :return: update bond info
        """

        self.info.constrain_mass = 3.0
        name = self.module_name + "_in_file"
        if name in self.controller.command_set:
            self.info.constrain_mass = 0
        if "mass" in self.controller.command_set:
            self.info.constrain_mass = float(self.controller.command_set["mass"])
        self.h_bond_pair = []
        s = 0
        for i in range(bond_numbers):
            mass_a = atom_mass[atom_a[i]]
            mass_b = atom_mass[atom_b[i]]
            if (float(mass_a) < self.info.constrain_mass and mass_a > 0) or \
                    (float(mass_b) < self.info.constrain_mass and mass_b > 0):
                constrain_k = \
                    atom_mass[atom_a[i]] * atom_mass[atom_b[i]] / (atom_mass[atom_a[i]] + atom_mass[atom_b[i]])
                self.h_bond_pair.append([atom_a[i], atom_b[i], bond_r[i], constrain_k])
                s += 1
        self.bond_constrain_pair_numbers = s
        self.bond_info.bond_numbers = bond_numbers
        self.bond_info.atom_a = atom_a
        self.bond_info.atom_b = atom_b
        self.bond_info.bond_r = bond_r

    def add_hangle_to_constrain_pair(self):
        """add hangle to constrain_pair"""

        self.h_angle_pair = []
        self.angle_constrain_pair_numbers = 0

    def initial_simple_constrain(self, atom_numbers, dt, box_length, exp_gamma, is_minimization, atom_mass,
                                 system_freedom):
        """initial simple constrain
        :param atom_numbers:
        :param dt:
        :param box_length:
        :param exp_gamma:
        :param is_minimization:
        :param atom_mass:
        :param system_freedom:
        :return:
        """

        self.system_freedom = system_freedom
        self.atom_mass = atom_mass
        self.info.atom_numbers = atom_numbers
        self.info.dt = dt
        self.info.quarter_crd_to_uint_crd_cof = [0.25 * self.constant_unit_max_float / box_length[0],
                                                 0.25 * self.constant_unit_max_float / box_length[1],
                                                 0.25 * self.constant_unit_max_float / box_length[2]]
        self.info.uint_dr_to_dr_cof = [1 / self.constant_unit_max_float * box_length[0],
                                       1 / self.constant_unit_max_float * box_length[1],
                                       1 / self.constant_unit_max_float * box_length[2]]
        self.info.volume = box_length[0] * box_length[1] * box_length[2]
        self.info.exp_gamma = exp_gamma
        self.half_exp_gamma_plus_half = 0.5 * (1. + self.info.exp_gamma)
        if is_minimization:
            self.info.exp_gamma = 0.0
        self.info.iteration_numbers = 25 if "iteration_numbers" not in self.controller.command_set else int(
            self.controller.command_set["iteration_numbers"])
        print("    constrain iteration step is ", self.info.iteration_numbers)
        self.info.step_length = 1 if "step_length" not in self.controller.command_set else float(
            self.controller.command_set["step_length"])
        print("    constrain step_length is ", self.info.step_length)
        self.extra_numbers = 0
        name = self.module_name + "_in_file"
        if name in self.controller.command_set:
            path = self.controller.command_set[name]
            with open(path, 'r') as pf:
                context = pf.readlines()
                self.extra_numbers = int(context[0].strip())

        self.dt_inverse = 1 / self.info.dt
        self.constrain_pair_numbers = \
            self.bond_constrain_pair_numbers + self.angle_constrain_pair_numbers + self.extra_numbers
        self.system_freedom -= self.constrain_pair_numbers

        print("    constrain pair number is ", self.constrain_pair_numbers)

        for i in range(self.bond_constrain_pair_numbers):
            self.h_constrain_pair.append(self.h_bond_pair[i])
            self.h_constrain_pair[i][-1] = self.info.step_length / self.half_exp_gamma_plus_half \
                * self.h_constrain_pair[i][-1]
        for i in range(self.angle_constrain_pair_numbers):
            self.h_constrain_pair.append(self.h_bond_pair[i])
            self.h_constrain_pair[i + self.bond_constrain_pair_numbers][-1] =\
                self.info.step_length / self.half_exp_gamma_plus_half * \
                self.h_constrain_pair[i + self.bond_constrain_pair_numbers][-1]
        if name in self.controller.command_set:
            path = self.controller.command_set[name]
            self.read_in_file(path)

    def read_in_file(self, path):
        """ read in_file

        :param path:
        :return: in_file context
        """

        with open(path, 'r') as pf:
            context = pf.readlines()
            count = self.bond_constrain_pair_numbers + self.angle_constrain_pair_numbers
            for i in range(self.extra_numbers):
                val = list(map(float, context[i + 1].strip().split()))
                atom_i, atom_j, constant_r = int(val[0]), int(val[1]), float(val[2])
                constrain_k = self.info.step_length / self.half_exp_gamma_plus_half * self.atom_mass[atom_i] * \
                    self.atom_mass[atom_j] / (self.atom_mass[atom_i] + self.atom_mass[atom_j])
                self.h_constrain_pair.append([atom_i, atom_j, constant_r, constrain_k])
                count += 1
