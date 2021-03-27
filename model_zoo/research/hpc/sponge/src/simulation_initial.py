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
"""simulation"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn

from .Langevin_Liujian_md import Langevin_Liujian
from .angle import Angle
from .bond import Bond
from .dihedral import Dihedral
from .lennard_jones import Lennard_Jones_Information
from .md_information import md_information
from .nb14 import NON_BOND_14
from .neighbor_list import nb_infomation
from .particle_mesh_ewald import Particle_Mesh_Ewald


class controller:
    """class controller"""
    def __init__(self, args_opt):
        self.input_file = args_opt.i
        self.initial_coordinates_file = args_opt.c
        self.amber_parm = args_opt.amber_parm
        self.restrt = args_opt.r
        self.mdcrd = args_opt.x
        self.mdout = args_opt.o
        self.mdbox = args_opt.box

        self.Command_Set = {}
        self.md_task = None
        self.commands_from_in_file()

    def commands_from_in_file(self):
        """commands from in file"""
        file = open(self.input_file, 'r')
        context = file.readlines()
        file.close()
        self.md_task = context[0].strip()
        for val in context:
            if "=" in val:
                assert len(val.strip().split("=")) == 2
                flag, value = val.strip().split("=")
                value = value.replace(",", '')
                flag = flag.replace(" ", "")
                if flag not in self.Command_Set:
                    self.Command_Set[flag] = value
                else:
                    print("ERROR COMMAND FILE")


class Simulation(nn.Cell):
    """class simulation"""

    def __init__(self, args_opt):
        super(Simulation, self).__init__()
        self.control = controller(args_opt)
        self.md_info = md_information(self.control)
        self.bond = Bond(self.control, self.md_info)
        self.angle = Angle(self.control)
        self.dihedral = Dihedral(self.control)
        self.nb14 = NON_BOND_14(self.control, self.dihedral, self.md_info.atom_numbers)
        self.nb_info = nb_infomation(self.control, self.md_info.atom_numbers, self.md_info.box_length)
        self.LJ_info = Lennard_Jones_Information(self.control)
        self.liujian_info = Langevin_Liujian(self.control, self.md_info.atom_numbers)
        self.pme_method = Particle_Mesh_Ewald(self.control, self.md_info)
        self.box_length = Tensor(np.asarray(self.md_info.box_length, np.float32), mstype.float32)
        self.file = None

    def Main_Before_Calculate_Force(self):
        """Main Before Calculate Force"""
        _ = self.md_info.MD_Information_Crd_To_Uint_Crd()
        self.md_info.uint_crd_with_LJ = (self.md_info.uint_crd, self.LJ_info.atom_LJ_type, self.md_info.charge)
        return self.md_info.uint_crd, self.md_info.uint_crd_with_LJ

    def Initial_Neighbor_List_Update(self, not_first_time):
        """Initial Neighbor List Update"""
        res = self.nb_info.NeighborListUpdate(self.md_info.crd, self.md_info.crd_old, self.md_info.uint_crd,
                                              self.md_info.crd_to_uint_crd_cof, self.md_info.uint_dr_to_dr_cof,
                                              self.box_length, not_first_time)

        return res

    def Main_Calculate_Force(self):
        """main calculate force"""
        self.bond.atom_numbers = self.md_info.atom_numbers
        md_info = self.md_info
        LJ_info = self.LJ_info
        nb_info = self.nb_info
        pme_method = self.pme_method
        bond_frc, _ = self.bond.Bond_Force_With_Atom_Energy(md_info.uint_crd, md_info.uint_dr_to_dr_cof)
        frc_t = 0
        frc_t += bond_frc.asnumpy()

        angle_frc, _ = self.angle.Angle_Force_With_Atom_Energy(md_info.uint_crd, md_info.uint_dr_to_dr_cof)
        frc_t += angle_frc.asnumpy()

        dihedral_frc, _ = self.dihedral.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.uint_dr_to_dr_cof)
        frc_t += dihedral_frc.asnumpy()

        nb14_frc, _ = self.nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy(md_info.uint_crd_with_LJ,
                                                                         md_info.uint_dr_to_dr_cof, LJ_info.LJ_A,
                                                                         LJ_info.LJ_B)
        frc_t += nb14_frc.asnumpy()

        lj_frc = LJ_info.LJ_Force_With_PME_Direct_Force(
            md_info.atom_numbers, md_info.uint_crd_with_LJ, md_info.uint_dr_to_dr_cof, nb_info.nl_atom_numbers,
            nb_info.nl_atom_serial, nb_info.cutoff, pme_method.beta)
        frc_t += lj_frc.asnumpy()

        pme_excluded_frc = pme_method.PME_Excluded_Force(
            md_info.uint_crd, md_info.uint_dr_to_dr_cof, md_info.charge,
            nb_info.excluded_list_start, nb_info.excluded_list,
            nb_info.excluded_numbers, nb_info.excluded_atom_numbers)
        frc_t += pme_excluded_frc.asnumpy()

        pme_reciprocal_frc = pme_method.PME_Reciprocal_Force(md_info.uint_crd, md_info.charge)
        frc_t += pme_reciprocal_frc.asnumpy()

        self.md_info.frc = Tensor(frc_t, mstype.float32)

        return self.md_info.frc

    def Main_Calculate_Energy(self):
        """main calculate energy"""
        _ = self.bond.Bond_Energy(self.md_info.uint_crd, self.md_info.uint_dr_to_dr_cof)
        _ = self.angle.Angle_Energy(self.md_info.uint_crd, self.md_info.uint_dr_to_dr_cof)
        _ = self.dihedral.Dihedral_Engergy(self.md_info.uint_crd, self.md_info.uint_dr_to_dr_cof)
        _ = self.nb14.Non_Bond_14_LJ_CF_Energy(self.md_info.uint_crd_with_LJ, self.md_info.uint_dr_to_dr_cof,
                                               self.LJ_info.LJ_A,
                                               self.LJ_info.LJ_B)

        _ = self.LJ_info.LJ_Energy(self.md_info.uint_crd_with_LJ, self.md_info.uint_dr_to_dr_cof,
                                   self.nb_info.nl_atom_numbers, self.nb_info.nl_atom_serial,
                                   self.nb_info.cutoff_square)
        _ = self.pme_method.PME_Energy(
            self.md_info.uint_crd, self.md_info.charge, self.nb_info.nl_atom_numbers, self.nb_info.nl_atom_serial,
            self.md_info.uint_dr_to_dr_cof, self.nb_info.excluded_list_start, self.nb_info.excluded_list,
            self.nb_info.excluded_numbers, self.nb_info.excluded_atom_numbers)
        _ = self.pme_method.Energy_Device_To_Host()

    def Main_After_Calculate_Energy(self):
        """main after calculate energy"""
        md_info = self.md_info
        LJ_info = self.LJ_info
        bond = self.bond
        angle = self.angle
        dihedral = self.dihedral
        nb14 = self.nb14
        pme_method = self.pme_method

        md_info.total_potential_energy = 0
        md_info.total_potential_energy += bond.sigma_of_bond_ene
        md_info.total_potential_energy += angle.sigma_of_angle_ene
        md_info.total_potential_energy += dihedral.sigma_of_dihedral_ene
        md_info.total_potential_energy += nb14.nb14_lj_energy_sum + nb14.nb14_cf_energy_sum
        md_info.total_potential_energy += LJ_info.LJ_energy_sum
        pme_method.Energy_Device_To_Host()
        md_info.total_potential_energy += pme_method.ee_ene
        print("md_info.total_potential_energy", md_info.total_potential_energy)

    def Main_Iteration_2(self):
        """main iteration2"""
        md_info = self.md_info
        control = self.control
        liujian_info = self.liujian_info

        if md_info.mode > 0 and int(control.Command_Set["thermostat"]) == 1:
            md_info.vel, md_info.crd, md_info.frc, md_info.acc = liujian_info.MD_Iteration_Leap_Frog(
                md_info.d_mass_inverse, md_info.vel, md_info.crd, md_info.frc)
            self.Main_After_Iteration()

    def Main_After_Iteration(self):
        """main after iteration"""
        md_info = self.md_info
        nb_info = self.nb_info
        md_info.Centerize()
        _ = nb_info.NeighborListUpdate(md_info.crd, md_info.crd_old, md_info.uint_crd,
                                       md_info.crd_to_uint_crd_cof,
                                       md_info.uint_dr_to_dr_cof, self.box_length, not_first_time=1)

    def Main_Print(self):
        """compute the temperature"""
        md_info = self.md_info
        temperature = md_info.MD_Information_Temperature()
        md_info.h_temperature = temperature
        steps = md_info.steps
        temperature = temperature.asnumpy()
        total_potential_energy = md_info.total_potential_energy.asnumpy()
        sigma_of_bond_ene = self.bond.sigma_of_bond_ene.asnumpy()
        sigma_of_angle_ene = self.angle.sigma_of_angle_ene.asnumpy()
        sigma_of_dihedral_ene = self.dihedral.sigma_of_dihedral_ene.asnumpy()
        nb14_lj_energy_sum = self.nb14.nb14_lj_energy_sum.asnumpy()
        nb14_cf_energy_sum = self.nb14.nb14_cf_energy_sum.asnumpy()
        LJ_energy_sum = self.LJ_info.LJ_energy_sum.asnumpy()
        ee_ene = self.pme_method.ee_ene.asnumpy()
        print("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
              "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_")
        print("{:>7.0f} {:>7.3f} {:>11.3f}".format(steps, float(temperature), float(total_potential_energy)), end=" ")
        if self.bond.bond_numbers > 0:
            print("{:>10.3f}".format(float(sigma_of_bond_ene)), end=" ")
        if self.angle.angle_numbers > 0:
            print("{:>11.3f}".format(float(sigma_of_angle_ene)), end=" ")
        if self.dihedral.dihedral_numbers > 0:
            print("{:>14.3f}".format(float(sigma_of_dihedral_ene)), end=" ")
        if self.nb14.nb14_numbers > 0:
            print("{:>10.3f} {:>10.3f}".format(float(nb14_lj_energy_sum), float(nb14_cf_energy_sum)), end=" ")

        print("{:>7.3f}".format(float(LJ_energy_sum)), end=" ")
        print("{:>12.3f}".format(float(ee_ene)))

        if self.file is not None:
            self.file.write("{:>7.0f} {:>7.3f} {:>11.3f} {:>10.3f} {:>11.3f} {:>14.3f} {:>10.3f} {:>10.3f} {:>7.3f}"
                            " {:>12.3f}\n".format(steps, float(temperature), float(total_potential_energy),
                                                  float(sigma_of_bond_ene), float(sigma_of_angle_ene),
                                                  float(sigma_of_dihedral_ene), float(nb14_lj_energy_sum),
                                                  float(nb14_cf_energy_sum), float(LJ_energy_sum), float(ee_ene)))

        return temperature

    def Main_Initial(self):
        """main initial"""
        if self.control.mdout:
            self.file = open(self.control.mdout, 'w')
            self.file.write("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                            "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_\n")

    def Main_Destroy(self):
        """main destroy"""
        if self.file is not None:
            self.file.close()
            print("Save successfully!")
