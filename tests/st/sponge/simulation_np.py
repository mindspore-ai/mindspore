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
'''Simulation'''
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from tests.st.sponge.sponge_cuda.angle import Angle
from tests.st.sponge.sponge_cuda.bond import Bond
from tests.st.sponge.sponge_cuda.dihedral import Dihedral
from tests.st.sponge.sponge_cuda.langevin_liujian_md import LangevinLiujian
from tests.st.sponge.sponge_cuda.lennard_jones import LennardJonesInformation
from tests.st.sponge.sponge_cuda.md_information import MdInformation
from tests.st.sponge.sponge_cuda.nb14 import NonBond14
from tests.st.sponge.sponge_cuda.neighbor_list import NeighborList
from tests.st.sponge.sponge_cuda.particle_mesh_ewald import ParticleMeshEwald
from tests.st.sponge.functions.dihedral_14_lj_energy import nb14_lj_energy
from tests.st.sponge.functions.dihedral_14_cf_energy import nb14_cf_energy
from tests.st.sponge.functions.common import reform_excluded_list, reform_residual_list, get_pme_bc
from tests.st.sponge.functions.angle_energy import angle_energy
from tests.st.sponge.functions.angle_force_with_atom_energy import angle_force_with_atom_energy
from tests.st.sponge.functions.bond_force_with_atom_energy import bond_force_with_atom_energy
from tests.st.sponge.functions.crd_to_uint_crd import crd_to_uint_crd
from tests.st.sponge.functions.dihedral_14_ljcf_force_with_atom_energy import dihedral_14_ljcf_force_with_atom_energy
from tests.st.sponge.functions.lj_energy import lj_energy
from tests.st.sponge.functions.md_iteration_leap_frog_liujian import md_iteration_leap_frog_liujian
from tests.st.sponge.functions.pme_excluded_force import pme_excluded_force
from tests.st.sponge.functions.lj_force_pme_direct_force import lj_force_pme_direct_force
from tests.st.sponge.functions.dihedral_force_with_atom_energy import dihedral_force_with_atom_energy
from tests.st.sponge.functions.bond_energy import bond_energy
from tests.st.sponge.functions.md_temperature import md_temperature
from tests.st.sponge.functions.dihedral_energy import dihedral_energy
from tests.st.sponge.functions.pme_energy import pme_energy
from tests.st.sponge.functions.pme_reciprocal_force import pme_reciprocal_force


class Controller:
    '''controller'''

    def __init__(self):
        self.input_file = "./data/NVT_290_10ns.in"
        self.amber_parm = "./data/WATER_ALA.parm7"
        self.initial_coordinates_file = "./data/WATER_ALA_350_cool_290.rst7"
        self.command_set = {}
        self.md_task = None
        self.commands_from_in_file()

    def commands_from_in_file(self):
        '''command from in file'''
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
                if flag not in self.command_set:
                    self.command_set[flag] = value
                else:
                    print("ERROR COMMAND FILE")


class Simulation(nn.Cell):
    '''simulation'''

    def __init__(self):
        super(Simulation, self).__init__()
        self.control = Controller()
        self.md_info = MdInformation(self.control)
        self.bond = Bond(self.control)
        self.angle = Angle(self.control)
        self.dihedral = Dihedral(self.control)
        self.nb14 = NonBond14(self.control, self.dihedral, self.md_info.atom_numbers)
        self.nb_info = NeighborList(self.control, self.md_info.atom_numbers, self.md_info.box_length)
        self.lj_info = LennardJonesInformation(self.control, self.md_info.nb.cutoff, self.md_info.sys.box_length)
        self.liujian_info = LangevinLiujian(self.control, self.md_info.atom_numbers)
        self.pme_method = ParticleMeshEwald(self.control, self.md_info)
        self.bond_energy_sum = Tensor(0, mstype.int32)
        self.angle_energy_sum = Tensor(0, mstype.int32)
        self.dihedral_energy_sum = Tensor(0, mstype.int32)
        self.nb14_lj_energy_sum = Tensor(0, mstype.int32)
        self.nb14_cf_energy_sum = Tensor(0, mstype.int32)
        self.lj_energy_sum = Tensor(0, mstype.int32)
        self.ee_ene = Tensor(0, mstype.int32)
        self.total_energy = Tensor(0, mstype.int32)
        # Init scalar
        self.ntwx = self.md_info.ntwx
        self.atom_numbers = self.md_info.atom_numbers
        self.residue_numbers = self.md_info.residue_numbers
        self.bond_numbers = self.bond.bond_numbers
        self.angle_numbers = self.angle.angle_numbers
        self.dihedral_numbers = self.dihedral.dihedral_numbers
        self.nb14_numbers = self.nb14.nb14_numbers
        self.nxy = self.nb_info.nxy
        self.grid_numbers = self.nb_info.grid_numbers
        self.max_atom_in_grid_numbers = self.nb_info.max_atom_in_grid_numbers
        self.max_neighbor_numbers = self.nb_info.max_neighbor_numbers
        self.excluded_atom_numbers = self.md_info.nb.excluded_atom_numbers
        self.refresh_count = Parameter(Tensor(self.nb_info.refresh_count, mstype.int32), requires_grad=False)
        self.refresh_interval = self.nb_info.refresh_interval
        self.skin = self.nb_info.skin
        self.cutoff = self.nb_info.cutoff
        self.cutoff_square = self.nb_info.cutoff_square
        self.cutoff_with_skin = self.nb_info.cutoff_with_skin
        self.half_cutoff_with_skin = self.nb_info.half_cutoff_with_skin
        self.cutoff_with_skin_square = self.nb_info.cutoff_with_skin_square
        self.half_skin_square = self.nb_info.half_skin_square
        self.beta = self.pme_method.beta
        self.fftx = self.pme_method.fftx
        self.ffty = self.pme_method.ffty
        self.fftz = self.pme_method.fftz
        self.box_length_0 = self.md_info.box_length[0]
        self.box_length_1 = self.md_info.box_length[1]
        self.box_length_2 = self.md_info.box_length[2]
        self.random_seed = self.liujian_info.random_seed
        self.dt = self.liujian_info.dt
        self.half_dt = self.liujian_info.half_dt
        self.exp_gamma = self.liujian_info.exp_gamma
        self.init_tensor()
        self.op_define()
        self.update = False

    def init_tensor(self):
        '''init tensor'''
        self.crd = Parameter(
            Tensor(np.float32(np.asarray(self.md_info.coordinate).reshape([self.atom_numbers, 3])), mstype.float32),
            requires_grad=False)
        self.crd_to_uint_crd_cof = Tensor(np.asarray(self.md_info.pbc.crd_to_uint_crd_cof, np.float32), mstype.float32)
        self.uint_dr_to_dr_cof = Parameter(
            Tensor(np.asarray(self.md_info.pbc.uint_dr_to_dr_cof, np.float32), mstype.float32), requires_grad=False)
        self.box_length = Tensor(self.md_info.box_length, mstype.float32)
        self.charge = Parameter(Tensor(np.asarray(self.md_info.h_charge, dtype=np.float32), mstype.float32),
                                requires_grad=False)
        self.old_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32),
                                 requires_grad=False)
        self.last_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32),
                                  requires_grad=False)
        self.uint_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.uint32), mstype.uint32),
                                  requires_grad=False)
        self.mass_inverse = Tensor(self.md_info.h_mass_inverse, mstype.float32)
        res_start = np.asarray(self.md_info.h_res_start, np.int32)
        res_end = np.asarray(self.md_info.h_res_end, np.int32)
        self.res_start = Tensor(res_start)
        self.res_end = Tensor(res_end)
        self.mass = Tensor(self.md_info.h_mass, mstype.float32)
        self.velocity = Parameter(Tensor(self.md_info.velocity, mstype.float32), requires_grad=False)
        self.acc = Parameter(Tensor(np.zeros([self.atom_numbers, 3], np.float32), mstype.float32), requires_grad=False)
        self.bond_atom_a = Tensor(np.asarray(self.bond.h_atom_a, np.int32), mstype.int32)
        self.bond_atom_b = Tensor(np.asarray(self.bond.h_atom_b, np.int32), mstype.int32)
        self.bond_k = Tensor(np.asarray(self.bond.h_k, np.float32), mstype.float32)
        self.bond_r0 = Tensor(np.asarray(self.bond.h_r0, np.float32), mstype.float32)
        self.angle_atom_a = Tensor(np.asarray(self.angle.h_atom_a, np.int32), mstype.int32)
        self.angle_atom_b = Tensor(np.asarray(self.angle.h_atom_b, np.int32), mstype.int32)
        self.angle_atom_c = Tensor(np.asarray(self.angle.h_atom_c, np.int32), mstype.int32)
        self.angle_k = Tensor(np.asarray(self.angle.h_angle_k, np.float32), mstype.float32)
        self.angle_theta0 = Tensor(np.asarray(self.angle.h_angle_theta0, np.float32), mstype.float32)
        self.dihedral_atom_a = Tensor(np.asarray(self.dihedral.h_atom_a, np.int32), mstype.int32)
        self.dihedral_atom_b = Tensor(np.asarray(self.dihedral.h_atom_b, np.int32), mstype.int32)
        self.dihedral_atom_c = Tensor(np.asarray(self.dihedral.h_atom_c, np.int32), mstype.int32)
        self.dihedral_atom_d = Tensor(np.asarray(self.dihedral.h_atom_d, np.int32), mstype.int32)
        self.pk = Tensor(np.asarray(self.dihedral.h_pk, np.float32), mstype.float32)
        self.gamc = Tensor(np.asarray(self.dihedral.h_gamc, np.float32), mstype.float32)
        self.gams = Tensor(np.asarray(self.dihedral.h_gams, np.float32), mstype.float32)
        self.pn = Tensor(np.asarray(self.dihedral.h_pn, np.float32), mstype.float32)
        self.ipn = Tensor(np.asarray(self.dihedral.h_ipn, np.int32), mstype.int32)
        self.nb14_atom_a = Tensor(np.asarray(self.nb14.h_atom_a, np.int32), mstype.int32)
        self.nb14_atom_b = Tensor(np.asarray(self.nb14.h_atom_b, np.int32), mstype.int32)
        self.lj_scale_factor = Tensor(np.asarray(self.nb14.h_lj_scale_factor, np.float32), mstype.float32)
        self.cf_scale_factor = Tensor(np.asarray(self.nb14.h_cf_scale_factor, np.float32), mstype.float32)
        self.grid_n = Tensor(self.nb_info.grid_n, mstype.int32)
        self.grid_length_inverse = Tensor(self.nb_info.grid_length_inverse, mstype.float32)
        self.bucket = Parameter(Tensor(
            np.asarray(self.nb_info.bucket, np.int32).reshape([self.grid_numbers, self.max_atom_in_grid_numbers]),
            mstype.int32), requires_grad=False)
        self.atom_numbers_in_grid_bucket = Parameter(Tensor(self.nb_info.atom_numbers_in_grid_bucket, mstype.int32),
                                                     requires_grad=False)
        self.atom_in_grid_serial = Parameter(Tensor(np.zeros([self.nb_info.atom_numbers], np.int32), mstype.int32),
                                             requires_grad=False)
        self.pointer = Parameter(
            Tensor(np.asarray(self.nb_info.pointer, np.int32).reshape([self.grid_numbers, 125]), mstype.int32),
            requires_grad=False)
        self.nl_atom_numbers = Parameter(Tensor(np.zeros([self.atom_numbers], np.int32), mstype.int32),
                                         requires_grad=False)
        self.nl_atom_serial = Parameter(
            Tensor(np.zeros([self.atom_numbers, self.max_neighbor_numbers], np.int32), mstype.int32),
            requires_grad=False)
        excluded_list_start = np.asarray(self.md_info.nb.h_excluded_list_start, np.int32)
        excluded_list = np.asarray(self.md_info.nb.h_excluded_list, np.int32)
        excluded_numbers = np.asarray(self.md_info.nb.h_excluded_numbers, np.int32)
        self.excluded_list_start = Tensor(excluded_list_start)
        self.excluded_list = Tensor(excluded_list)
        self.excluded_numbers = Tensor(excluded_numbers)
        self.excluded_matrix = Tensor(reform_excluded_list(excluded_list, excluded_list_start, excluded_numbers))
        self.residual_matrix = Tensor(reform_residual_list(self.atom_numbers, res_start, res_end))
        box = (self.box_length_0, self.box_length_1, self.box_length_2)
        self.pme_bc = Tensor(get_pme_bc(self.fftx, self.ffty, self.fftz, box, self.beta), mstype.float32)
        self.need_refresh_flag = Tensor(np.asarray([0], np.int32), mstype.int32)
        self.atom_lj_type = Tensor(self.lj_info.atom_lj_type, mstype.int32)
        self.lj_a = Tensor(self.lj_info.h_lj_a, mstype.float32)
        self.lj_b = Tensor(self.lj_info.h_lj_b, mstype.float32)
        self.sqrt_mass = Tensor(self.liujian_info.h_sqrt_mass, mstype.float32)
        self.rand_state = Parameter(Tensor(self.liujian_info.rand_state, mstype.float32))
        self.zero_fp_tensor = Tensor(np.asarray([0], np.float32))

    def op_define(self):
        '''op define'''
        self.neighbor_list_update_init = P.NeighborListUpdate(grid_numbers=self.grid_numbers,
                                                              atom_numbers=self.atom_numbers, not_first_time=0,
                                                              nxy=self.nxy,
                                                              excluded_atom_numbers=self.excluded_atom_numbers,
                                                              cutoff_square=self.cutoff_square,
                                                              half_skin_square=self.half_skin_square,
                                                              cutoff_with_skin=self.cutoff_with_skin,
                                                              half_cutoff_with_skin=self.half_cutoff_with_skin,
                                                              cutoff_with_skin_square=self.cutoff_with_skin_square,
                                                              refresh_interval=self.refresh_interval,
                                                              cutoff=self.cutoff, skin=self.skin,
                                                              max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                                              max_neighbor_numbers=self.max_neighbor_numbers)
        self.neighbor_list_update = P.NeighborListUpdate(grid_numbers=self.grid_numbers, atom_numbers=self.atom_numbers,
                                                         not_first_time=1, nxy=self.nxy,
                                                         excluded_atom_numbers=self.excluded_atom_numbers,
                                                         cutoff_square=self.cutoff_square,
                                                         half_skin_square=self.half_skin_square,
                                                         cutoff_with_skin=self.cutoff_with_skin,
                                                         half_cutoff_with_skin=self.half_cutoff_with_skin,
                                                         cutoff_with_skin_square=self.cutoff_with_skin_square,
                                                         refresh_interval=self.refresh_interval, cutoff=self.cutoff,
                                                         skin=self.skin,
                                                         max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                                         max_neighbor_numbers=self.max_neighbor_numbers)

    def simulation_beforce_caculate_force(self):
        '''simulation before calculate force'''
        crd_to_uint_crd_cof = 0.5 * self.crd_to_uint_crd_cof
        uint_crd = crd_to_uint_crd(crd_to_uint_crd_cof, self.crd)
        return uint_crd

    def simulation_caculate_force(self, uint_crd, scaler, nl_atom_numbers, nl_atom_serial):
        '''simulation calculate force'''
        bond_f, _ = bond_force_with_atom_energy(self.atom_numbers, self.bond_numbers,
                                                uint_crd, scaler, self.bond_atom_a,
                                                self.bond_atom_b, self.bond_k,
                                                self.bond_r0)

        angle_f, _ = angle_force_with_atom_energy(self.angle_numbers, uint_crd,
                                                  scaler, self.angle_atom_a,
                                                  self.angle_atom_b, self.angle_atom_c,
                                                  self.angle_k, self.angle_theta0)

        dihedral_f, _ = dihedral_force_with_atom_energy(self.dihedral_numbers,
                                                        uint_crd, scaler,
                                                        self.dihedral_atom_a,
                                                        self.dihedral_atom_b,
                                                        self.dihedral_atom_c,
                                                        self.dihedral_atom_d, self.ipn,
                                                        self.pk, self.gamc, self.gams,
                                                        self.pn)

        nb14_f, _ = dihedral_14_ljcf_force_with_atom_energy(self.atom_numbers, uint_crd,
                                                            self.atom_lj_type, self.charge,
                                                            scaler, self.nb14_atom_a, self.nb14_atom_b,
                                                            self.lj_scale_factor, self.cf_scale_factor,
                                                            self.lj_a, self.lj_b)

        lj_f = lj_force_pme_direct_force(self.atom_numbers, self.cutoff_square, self.beta,
                                         uint_crd, self.atom_lj_type, self.charge, scaler,
                                         nl_atom_numbers, nl_atom_serial, self.lj_a, self.lj_b)

        pme_excluded_f = pme_excluded_force(self.atom_numbers, self.beta, uint_crd, scaler,
                                            self.charge, excluded_matrix=self.excluded_matrix)

        pme_reciprocal_f = pme_reciprocal_force(self.atom_numbers, self.fftx, self.ffty,
                                                self.fftz, self.box_length_0, self.box_length_1,
                                                self.box_length_2, self.pme_bc, uint_crd, self.charge)
        force = bond_f + angle_f + dihedral_f + nb14_f + lj_f + pme_excluded_f + pme_reciprocal_f
        return force

    def simulation_caculate_energy(self, uint_crd, uint_dr_to_dr_cof):
        '''simulation calculate energy'''
        bond_e = bond_energy(self.atom_numbers, self.bond_numbers, uint_crd, uint_dr_to_dr_cof,
                             self.bond_atom_a, self.bond_atom_b, self.bond_k, self.bond_r0)
        bond_energy_sum = bond_e.sum(keepdims=True)

        angle_e = angle_energy(self.angle_numbers, uint_crd, uint_dr_to_dr_cof,
                               self.angle_atom_a, self.angle_atom_b, self.angle_atom_c,
                               self.angle_k, self.angle_theta0)
        angle_energy_sum = angle_e.sum(keepdims=True)

        dihedral_e = dihedral_energy(self.dihedral_numbers, uint_crd, uint_dr_to_dr_cof,
                                     self.dihedral_atom_a, self.dihedral_atom_b,
                                     self.dihedral_atom_c, self.dihedral_atom_d,
                                     self.ipn, self.pk, self.gamc, self.gams,
                                     self.pn)
        dihedral_energy_sum = dihedral_e.sum(keepdims=True)

        nb14_lj_e = nb14_lj_energy(self.nb14_numbers, self.atom_numbers, uint_crd,
                                   self.atom_lj_type, self.charge, uint_dr_to_dr_cof,
                                   self.nb14_atom_a, self.nb14_atom_b, self.lj_scale_factor, self.lj_a,
                                   self.lj_b)
        nb14_cf_e = nb14_cf_energy(self.nb14_numbers, self.atom_numbers, uint_crd,
                                   self.atom_lj_type, self.charge, uint_dr_to_dr_cof,
                                   self.nb14_atom_a, self.nb14_atom_b, self.cf_scale_factor)
        nb14_lj_energy_sum = nb14_lj_e.sum(keepdims=True)
        nb14_cf_energy_sum = nb14_cf_e.sum(keepdims=True)

        lj_e = lj_energy(self.atom_numbers, self.cutoff_square, uint_crd,
                         self.atom_lj_type, uint_dr_to_dr_cof,
                         self.nl_atom_numbers, self.nl_atom_serial, self.lj_a,
                         self.lj_b)
        lj_energy_sum = lj_e.sum(keepdims=True)
        reciprocal_e, self_e, direct_e, correction_e = pme_energy(self.atom_numbers,
                                                                  self.beta, self.fftx,
                                                                  self.ffty, self.fftz,
                                                                  self.pme_bc,
                                                                  uint_crd, self.charge,
                                                                  self.nl_atom_numbers,
                                                                  self.nl_atom_serial,
                                                                  uint_dr_to_dr_cof,
                                                                  excluded_matrix=self.excluded_matrix)

        ee_ene = reciprocal_e + self_e + direct_e + correction_e
        total_energy = bond_energy_sum + angle_energy_sum + dihedral_energy_sum + \
                       nb14_lj_energy_sum + nb14_cf_energy_sum + lj_energy_sum + ee_ene
        res_cpt_ene = (bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
               lj_energy_sum, ee_ene, total_energy)
        return res_cpt_ene

    def simulation_temperature(self):
        '''caculate temperature'''
        res_ek_energy = md_temperature(self.residual_matrix, self.velocity, self.mass, sparse=False)
        temperature = res_ek_energy.sum()
        return temperature

    def simulation_md_iteration_leap_frog_liujian(self, inverse_mass, sqrt_mass_inverse, crd, frc):
        '''simulation leap frog iteration liujian'''
        return md_iteration_leap_frog_liujian(self.atom_numbers, self.half_dt,
                                              self.dt, self.exp_gamma, inverse_mass,
                                              sqrt_mass_inverse, self.velocity,
                                              crd, frc, self.acc)

    def construct(self):
        '''construct'''
        self.last_crd = self.crd
        res = self.neighbor_list_update(self.atom_numbers_in_grid_bucket,
                                        self.bucket,
                                        self.crd,
                                        self.box_length,
                                        self.grid_n,
                                        self.grid_length_inverse,
                                        self.atom_in_grid_serial,
                                        self.old_crd,
                                        self.crd_to_uint_crd_cof,
                                        self.uint_crd,
                                        self.pointer,
                                        self.nl_atom_numbers,
                                        self.nl_atom_serial,
                                        self.uint_dr_to_dr_cof,
                                        self.excluded_list_start,
                                        self.excluded_list,
                                        self.excluded_numbers,
                                        self.need_refresh_flag,
                                        self.refresh_count)
        self.nl_atom_numbers = F.depend(self.nl_atom_numbers, res)
        self.nl_atom_serial = F.depend(self.nl_atom_serial, res)
        self.uint_dr_to_dr_cof = F.depend(self.uint_dr_to_dr_cof, res)
        self.old_crd = F.depend(self.old_crd, res)
        self.atom_numbers_in_grid_bucket = F.depend(self.atom_numbers_in_grid_bucket, res)
        self.bucket = F.depend(self.bucket, res)
        self.atom_in_grid_serial = F.depend(self.atom_in_grid_serial, res)
        self.pointer = F.depend(self.pointer, res)
        uint_crd = self.simulation_beforce_caculate_force()
        force = self.simulation_caculate_force(uint_crd, self.uint_dr_to_dr_cof, self.nl_atom_numbers,
                                               self.nl_atom_serial)

        bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
        lj_energy_sum, ee_ene, total_energy = self.simulation_caculate_energy(uint_crd, self.uint_dr_to_dr_cof)

        temperature = self.simulation_temperature()
        self.velocity = F.depend(self.velocity, temperature)
        self.velocity, self.crd, _ = self.simulation_md_iteration_leap_frog_liujian(self.mass_inverse,
                                                                                    self.sqrt_mass, self.crd, force)
        res_main = (temperature, total_energy, bond_energy_sum, angle_energy_sum, dihedral_energy_sum, \
            nb14_lj_energy_sum, nb14_cf_energy_sum, lj_energy_sum, ee_ene, res)
        return res_main
