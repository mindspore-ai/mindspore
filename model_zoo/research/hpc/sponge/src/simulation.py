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
from src.angle import Angle
from src.bd_baro import BD_BARO
from src.bond import Bond
from src.crd_molecular_map import CoordinateMolecularMap
from src.dihedral import Dihedral
from src.langevin_liujian_md import Langevin_Liujian
from src.lennard_jones import Lennard_Jones_Information
from src.mc_baro import MC_BARO
from src.md_information import md_information
from src.nb14 import NON_BOND_14
from src.neighbor_list import neighbor_list
from src.particle_mesh_ewald import Particle_Mesh_Ewald
from src.restrain import Restrain_Information
from src.simple_constrain import Simple_Constarin
from src.vatom import Virtual_Information

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class controller:
    '''controller'''

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
        self.punctuation = ","

    def commands_from_in_file(self):
        '''command from in file'''
        file = open(self.input_file, 'r')
        context = file.readlines()
        file.close()
        self.md_task = context[0].strip()
        for val in context:
            val = val.strip()
            if val and val[0] != '#' and ("=" in val):
                val = val[:val.index(",")] if ',' in val else val
                assert len(val.strip().split("=")) == 2
                flag, value = val.strip().split("=")
                value = value.replace(" ", "")
                flag = flag.replace(" ", "")
                if flag not in self.Command_Set:
                    self.Command_Set[flag] = value
                else:
                    print("ERROR COMMAND FILE")


class Simulation(nn.Cell):
    '''simulation'''

    def __init__(self, args_opt):
        super(Simulation, self).__init__()
        self.control = controller(args_opt)
        self.md_info = md_information(self.control)
        self.mode = self.md_info.mode
        self.bond = Bond(self.control)
        self.bond_is_initialized = self.bond.is_initialized
        self.angle = Angle(self.control)
        self.angle_is_initialized = self.angle.is_initialized
        self.dihedral = Dihedral(self.control)
        self.dihedral_is_initialized = self.dihedral.is_initialized
        self.nb14 = NON_BOND_14(self.control, self.dihedral, self.md_info.atom_numbers)
        self.nb14_is_initialized = self.nb14.is_initialized
        self.nb_info = neighbor_list(self.control, self.md_info.atom_numbers, self.md_info.box_length)
        self.LJ_info = Lennard_Jones_Information(self.control, self.md_info.nb.cutoff, self.md_info.sys.box_length)
        self.LJ_info_is_initialized = self.LJ_info.is_initialized

        self.liujian_info = Langevin_Liujian(self.control, self.md_info.atom_numbers)
        self.liujian_info_is_initialized = self.liujian_info.is_initialized
        self.pme_method = Particle_Mesh_Ewald(self.control, self.md_info)
        self.pme_is_initialized = self.pme_method.is_initialized
        self.restrain = Restrain_Information(self.control, self.md_info.atom_numbers, self.md_info.crd)
        self.restrain_is_initialized = self.restrain.is_initialized
        self.simple_constrain_is_initialized = 0

        self.simple_constrain = Simple_Constarin(self.control, self.md_info, self.bond, self.angle, self.liujian_info)
        self.simple_constrain_is_initialized = self.simple_constrain.is_initialized
        self.freedom = self.simple_constrain.system_freedom

        self.vatom = Virtual_Information(self.control, self.md_info, self.md_info.sys.freedom)
        self.vatom_is_initialized = 1

        self.random = P.UniformReal(seed=1)
        self.pow = P.Pow()

        self.mol_map = CoordinateMolecularMap(self.md_info.atom_numbers, self.md_info.sys.box_length, self.md_info.crd,
                                              self.md_info.nb.excluded_atom_numbers, self.md_info.nb.h_excluded_numbers,
                                              self.md_info.nb.h_excluded_list_start, self.md_info.nb.h_excluded_list)
        self.mol_map_is_initialized = 1
        self.init_params()
        self.init_Tensor()
        self.op_define()
        self.op_define_2()
        self.depend = P.Depend()
        self.print = P.Print()
        self.total_count = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.accept_count = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.is_molecule_map_output = self.md_info.output.is_molecule_map_output
        self.target_pressure = self.md_info.sys.target_pressure
        self.Nx = self.nb_info.Nx
        self.Ny = self.nb_info.Ny
        self.Nz = self.nb_info.Nz
        self.PME_inverse_box_vector = Parameter(Tensor(self.pme_method.PME_inverse_box_vector, mstype.float32),
                                                requires_grad=False)
        self.mc_baro_is_initialized = 0
        self.bd_baro_is_initialized = 0

        if self.mode == 2 and self.control.Command_Set["barostat"] == "monte_carlo":
            self.mc_baro = MC_BARO(self.control, self.md_info.atom_numbers, self.md_info.sys.target_pressure,
                                   self.md_info.sys.box_length, self.md_info.res.is_initialized, self.md_info.mode)
            self.mc_baro_is_initialized = self.mc_baro.is_initialized
            self.update_interval = self.mc_baro.update_interval
            self.mc_baro_energy_old = Parameter(Tensor(0, mstype.float32), requires_grad=False)
            self.potential = Parameter(Tensor(0, mstype.float32), requires_grad=False)
            self.frc_backup = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)
            self.crd_backup = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)
            self.crd_scale_factor = Parameter(Tensor(0.0, mstype.float32), requires_grad=False)
            self.system_reinitializing_count = Parameter(Tensor(0, mstype.int32), requires_grad=False)
            self.mc_baro_energy_new = Parameter(Tensor(0.0, mstype.float32), requires_grad=False)
            self.scale_coordinate_by_residue = Parameter(Tensor(0, mstype.float32), requires_grad=False)
            self.extra_term = Parameter(Tensor(0, mstype.float32), requires_grad=False)
            self.DeltaV = Parameter(Tensor(0.0, mstype.float32), requires_grad=False)
            self.target_temperature = self.md_info.sys.target_temperature
            self.VDevided = Parameter(Tensor(0.0, mstype.float32), requires_grad=False)
            self.log = P.Log()
            self.mc_baro_accept_possibility = Parameter(Tensor(0, mstype.float32), requires_grad=False)
            self.exp = P.Exp()
            self.mc_baro_newV = self.mc_baro.newV
            self.mc_baro_V0 = Parameter(Tensor(self.mc_baro.V0, mstype.float32), requires_grad=False)
            self.mc_baro_newV = self.mc_baro.newV
            self.check_interval = self.mc_baro.check_interval

        if self.mode == 2 and self.control.Command_Set["barostat"] == "berendsen":
            self.bd_baro = BD_BARO(self.control, self.md_info.sys.target_pressure, self.md_info.sys.box_length,
                                   self.md_info.mode)
            self.bd_baro_is_initialized = self.bd_baro.is_initialized
            self.update_interval = self.bd_baro.update_interval
            self.pressure = Parameter(Tensor(self.md_info.sys.d_pressure, mstype.float32), requires_grad=False)
            self.compressibility = self.bd_baro.compressibility
            self.bd_baro_dt = self.bd_baro.dt
            self.bd_baro_taup = self.bd_baro.taup
            self.system_reinitializing_count = Parameter(Tensor(0, mstype.int32), requires_grad=False)
            self.bd_baro_newV = Parameter(Tensor(self.bd_baro.newV, mstype.float32), requires_grad=False)
            self.bd_baro_V0 = Parameter(Tensor(self.bd_baro.V0, mstype.float32), requires_grad=False)

    def init_params(self):
        """init_params"""
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
        self.nxy = self.nb_info.Nxy
        self.grid_numbers = self.nb_info.grid_numbers
        self.max_atom_in_grid_numbers = self.nb_info.max_atom_in_grid_numbers
        self.max_neighbor_numbers = self.nb_info.max_neighbor_numbers
        # self.excluded_atom_numbers = self.nb_info.excluded_atom_numbers
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
        self.random_seed = self.liujian_info.random_seed
        self.dt = self.liujian_info.dt
        self.half_dt = self.liujian_info.half_dt
        self.exp_gamma = self.liujian_info.exp_gamma
        self.update = False
        self.file = None
        self.datfile = None
        self.max_velocity = self.liujian_info.max_velocity

        # bingshui
        self.CONSTANT_kB = 0.00198716

    def init_Tensor(self):
        '''init tensor'''
        # MD_Reset_Atom_Energy_And_Virial
        self.uint_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.uint32), mstype.uint32),
                                  requires_grad=False)
        self.need_potential = Tensor(0, mstype.int32)
        self.need_pressure = Tensor(0, mstype.int32)
        # self.potential = Tensor(0, mstype.float32)
        self.atom_energy = Parameter(Tensor([0] * self.atom_numbers, mstype.float32), requires_grad=False)
        self.atom_virial = Parameter(Tensor([0] * self.atom_numbers, mstype.float32), requires_grad=False)
        self.frc = Parameter(Tensor(np.zeros([self.atom_numbers, 3]), mstype.float32), requires_grad=False)

        self.crd = Parameter(
            Tensor(np.array(self.md_info.coordinate).reshape([self.atom_numbers, 3]), mstype.float32),
            requires_grad=False)
        self.crd_to_uint_crd_cof = Tensor(np.asarray(self.md_info.pbc.crd_to_uint_crd_cof, np.float32), mstype.float32)
        self.quarter_crd_to_uint_crd_cof = Tensor(np.asarray(self.md_info.pbc.quarter_crd_to_uint_crd_cof, np.float32),
                                                  mstype.float32)

        self.uint_dr_to_dr_cof = Parameter(Tensor(self.md_info.pbc.uint_dr_to_dr_cof, mstype.float32),
                                           requires_grad=False)
        self.box_length = Tensor(self.md_info.box_length, mstype.float32)
        self.charge = Parameter(Tensor(np.asarray(self.md_info.h_charge, dtype=np.float32), mstype.float32),
                                requires_grad=False)
        self.old_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32),
                                 requires_grad=False)
        self.last_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32),
                                  requires_grad=False)
        self.mass = Tensor(self.md_info.h_mass, mstype.float32)
        self.mass_inverse = Tensor(self.md_info.h_mass_inverse, mstype.float32)
        self.res_mass = Tensor(self.md_info.res.h_mass, mstype.float32)
        self.res_mass_inverse = Tensor(self.md_info.res.h_mass_inverse, mstype.float32)

        self.res_start = Tensor(self.md_info.h_res_start, mstype.int32)
        self.res_end = Tensor(self.md_info.h_res_end, mstype.int32)
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
        self.grid_N = Tensor(self.nb_info.grid_N, mstype.int32)
        self.grid_length = Parameter(Tensor(self.nb_info.grid_length, mstype.float32), requires_grad=False)
        self.grid_length_inverse = Parameter(Tensor(self.nb_info.grid_length_inverse, mstype.float32),
                                             requires_grad=False)
        self.bucket = Parameter(Tensor(
            np.asarray(self.nb_info.bucket, np.int32).reshape([self.grid_numbers, self.max_atom_in_grid_numbers]),
            mstype.int32), requires_grad=False)
        self.atom_numbers_in_grid_bucket = Parameter(Tensor(self.nb_info.atom_numbers_in_grid_bucket, mstype.int32),
                                                     requires_grad=False)
        self.atom_in_grid_serial = Parameter(Tensor(np.zeros([self.nb_info.atom_numbers,], np.int32), mstype.int32),
                                             requires_grad=False)
        self.pointer = Parameter(
            Tensor(np.asarray(self.nb_info.pointer, np.int32).reshape([self.grid_numbers, 125]), mstype.int32),
            requires_grad=False)
        self.nl_atom_numbers = Parameter(Tensor(np.zeros([self.atom_numbers,], np.int32), mstype.int32),
                                         requires_grad=False)
        self.nl_atom_serial = Parameter(
            Tensor(np.zeros([self.atom_numbers, self.max_neighbor_numbers], np.int32), mstype.int32),
            requires_grad=False)
        self.excluded_list_start = Tensor(np.asarray(self.md_info.nb.h_excluded_list_start, np.int32), mstype.int32)
        self.excluded_list = Tensor(np.asarray(self.md_info.nb.h_excluded_list, np.int32), mstype.int32)
        self.excluded_numbers = Tensor(np.asarray(self.md_info.nb.h_excluded_numbers, np.int32), mstype.int32)

        self.need_refresh_flag = Tensor(np.asarray([0], np.int32), mstype.int32)
        self.atom_LJ_type = Tensor(self.LJ_info.atom_LJ_type, mstype.int32)
        self.LJ_A = Tensor(self.LJ_info.h_LJ_A, mstype.float32)
        self.LJ_B = Tensor(self.LJ_info.h_LJ_B, mstype.float32)
        self.sqrt_mass = Tensor(self.liujian_info.h_sqrt_mass, mstype.float32)
        self.rand_state = Parameter(Tensor(self.liujian_info.rand_state, mstype.float32))
        self.zero_fp_tensor = Tensor(np.asarray([0,], np.float32))
        self.zero_frc = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32),
                                  requires_grad=False)

    def op_define(self):
        '''op define'''
        self.crd_to_uint_crd = P.CrdToUintCrd(self.atom_numbers)
        self.crd_to_uint_crd_quarter = P.CrdToUintCrdQuarter(self.atom_numbers)
        self.mdtemp = P.MDTemperature(self.residue_numbers, self.atom_numbers)
        self.setup_random_state = P.MDIterationSetupRandState(self.atom_numbers, self.random_seed)

        self.bond_force_with_atom_energy_virial = P.BondForceWithAtomEnergyAndVirial(bond_numbers=self.bond_numbers,
                                                                                     atom_numbers=self.atom_numbers)
        self.angle_force_with_atom_energy = P.AngleForceWithAtomEnergy(angle_numbers=self.angle_numbers)
        self.dihedral_force_with_atom_energy = P.DihedralForceWithAtomEnergy(dihedral_numbers=self.dihedral_numbers)
        self.nb14_force_with_atom_energy = P.Dihedral14LJCFForceWithAtomEnergy(nb14_numbers=self.nb14_numbers,
                                                                               atom_numbers=self.atom_numbers)
        self.lj_force_pme_direct_force = P.LJForceWithPMEDirectForce(self.atom_numbers, self.cutoff, self.beta)
        self.pme_excluded_force = P.PMEExcludedForce(atom_numbers=self.atom_numbers,
                                                     excluded_numbers=self.excluded_atom_numbers, beta=self.beta)
        self.pme_reciprocal_force = P.PMEReciprocalForce(self.atom_numbers, self.beta, self.fftx, self.ffty, self.fftz,
                                                         self.md_info.box_length[0], self.md_info.box_length[1],
                                                         self.md_info.box_length[2])
        self.bond_energy = P.BondEnergy(self.bond_numbers, self.atom_numbers)
        self.angle_energy = P.AngleEnergy(self.angle_numbers)
        self.dihedral_energy = P.DihedralEnergy(self.dihedral_numbers)
        self.nb14_lj_energy = P.Dihedral14LJEnergy(self.nb14_numbers, self.atom_numbers)
        self.nb14_cf_energy = P.Dihedral14CFEnergy(self.nb14_numbers, self.atom_numbers)
        self.lj_energy = P.LJEnergy(self.atom_numbers, self.cutoff_square)
        self.pme_energy = P.PMEEnergy(self.atom_numbers, self.excluded_atom_numbers, self.beta, self.fftx, self.ffty,
                                      self.fftz, self.md_info.box_length[0], self.md_info.box_length[1],
                                      self.md_info.box_length[2])
        self.md_iteration_leap_frog_liujian = P.MDIterationLeapFrogLiujian(self.atom_numbers, self.half_dt, self.dt,
                                                                           self.exp_gamma)

        self.md_iteration_leap_frog_liujian_with_max_vel = P.MDIterationLeapFrogLiujianWithMaxVel(self.atom_numbers,
                                                                                                  self.half_dt, self.dt,
                                                                                                  self.exp_gamma,
                                                                                                  self.max_velocity)
        self.neighbor_list_update = \
            P.NeighborListUpdate(grid_numbers=self.grid_numbers,
                                 atom_numbers=self.atom_numbers,
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

        self.neighbor_list_update_forced_update = \
            P.NeighborListUpdate(grid_numbers=self.grid_numbers,
                                 atom_numbers=self.atom_numbers,
                                 not_first_time=1, nxy=self.nxy,
                                 excluded_atom_numbers=self.excluded_atom_numbers,
                                 cutoff_square=self.cutoff_square,
                                 half_skin_square=self.half_skin_square,
                                 cutoff_with_skin=self.cutoff_with_skin,
                                 half_cutoff_with_skin=self.half_cutoff_with_skin,
                                 cutoff_with_skin_square=self.cutoff_with_skin_square,
                                 refresh_interval=self.refresh_interval,
                                 cutoff=self.cutoff,
                                 skin=self.skin,
                                 max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                 max_neighbor_numbers=self.max_neighbor_numbers,
                                 forced_update=1)

        self.neighbor_list_update_nb = \
            P.NeighborListUpdate(grid_numbers=self.grid_numbers,
                                 atom_numbers=self.atom_numbers,
                                 not_first_time=1, nxy=self.nxy,
                                 excluded_atom_numbers=self.excluded_atom_numbers,
                                 cutoff_square=self.cutoff_square,
                                 half_skin_square=self.half_skin_square,
                                 cutoff_with_skin=self.cutoff_with_skin,
                                 half_cutoff_with_skin=self.half_cutoff_with_skin,
                                 cutoff_with_skin_square=self.cutoff_with_skin_square,
                                 refresh_interval=self.refresh_interval,
                                 cutoff=self.cutoff,
                                 skin=self.skin,
                                 max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                 max_neighbor_numbers=self.max_neighbor_numbers,
                                 forced_update=1, forced_check=1)

    def op_define_2(self):
        """op_define_2"""
        self.neighbor_list_update_mc = P.NeighborListUpdate(grid_numbers=self.grid_numbers,
                                                            atom_numbers=self.atom_numbers,
                                                            not_first_time=1, nxy=self.nxy,
                                                            excluded_atom_numbers=self.excluded_atom_numbers,
                                                            cutoff_square=self.cutoff_square,
                                                            half_skin_square=self.half_skin_square,
                                                            cutoff_with_skin=self.cutoff_with_skin,
                                                            half_cutoff_with_skin=self.half_cutoff_with_skin,
                                                            cutoff_with_skin_square=self.cutoff_with_skin_square,
                                                            refresh_interval=self.refresh_interval,
                                                            cutoff=self.cutoff,
                                                            skin=self.skin,
                                                            max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                                            max_neighbor_numbers=self.max_neighbor_numbers,
                                                            forced_update=0, forced_check=1)

        self.random_force = Tensor(np.zeros([self.atom_numbers, 3], np.float32), mstype.float32)

        # simple_constrain
        self.constrain_pair_numbers = self.simple_constrain.constrain_pair_numbers
        self.last_pair_dr = Parameter(Tensor(np.zeros([self.constrain_pair_numbers, 3], np.float32), mstype.float32),
                                      requires_grad=False)
        if self.simple_constrain_is_initialized:
            self.constrain_pair_numbers = self.simple_constrain.constrain_pair_numbers
            self.last_crd_to_dr = P.lastcrdtodr(self.atom_numbers, self.constrain_pair_numbers)
            self.constrain_pair = np.array(self.simple_constrain.h_constrain_pair)
            self.atom_i_serials = Tensor(self.constrain_pair[:, 0], mstype.int32)
            self.atom_j_serials = Tensor(self.constrain_pair[:, 1], mstype.int32)
            self.constant_rs = Tensor(self.constrain_pair[:, 2], mstype.float32)
            self.constrain_ks = Tensor(self.constrain_pair[:, 3], mstype.float32)
            self.last_pair_dr = Parameter(
                Tensor(np.zeros([self.constrain_pair_numbers, 3], np.float32), mstype.float32), requires_grad=False)
            self.constrain_frc = Parameter(Tensor(np.zeros([self.atom_numbers, 3], np.float32), mstype.float32),
                                           requires_grad=False)
            self.iteration_numbers = self.simple_constrain.info.iteration_numbers
            self.half_exp_gamma_plus_half = self.simple_constrain.half_exp_gamma_plus_half
            self.refresh_uint_crd = P.refreshuintcrd(self.atom_numbers, self.half_exp_gamma_plus_half)
            self.need_pressure = 0
            self.constrain_force_cycle_with_virial = P.constrainforcecyclewithvirial(self.atom_numbers,
                                                                                     self.constrain_pair_numbers)
            self.constrain_force_cycle = P.ConstrainForceCycle(self.atom_numbers, self.constrain_pair_numbers)
            self.dt_inverse = self.simple_constrain.dt_inverse
            self.refresh_crd_vel = P.refreshcrdvel(self.atom_numbers, self.dt_inverse, self.dt, self.exp_gamma,
                                                   self.half_exp_gamma_plus_half)

        if self.mol_map_is_initialized:
            self.refresh_boxmaptimes = P.refreshboxmaptimes(self.atom_numbers)
            self.box_map_times = Parameter(Tensor(self.mol_map.h_box_map_times, mstype.int32), requires_grad=False)
        self.residue_numbers = self.md_info.residue_numbers
        self.getcenterofmass = P.GetCenterOfMass(self.residue_numbers)
        self.mapcenterofmass = P.MapCenterOfMass(self.residue_numbers, scaler=1.0)

        self.md_iteration_leap_frog = P.MDIterationLeapFrog(self.atom_numbers, self.dt)
        self.md_iteration_leap_frog_with_max_vel = P.MDIterationLeapFrogWithMaxVel(self.atom_numbers, self.dt,
                                                                                   self.max_velocity)
        self.md_information_gradient_descent = P.MDIterationGradientDescent(self.atom_numbers, self.dt * self.dt)

    def Simulation_Beforce_Caculate_Force(self):
        '''simulation before calculate force'''
        self.uint_crd = self.crd_to_uint_crd_quarter(self.quarter_crd_to_uint_crd_cof, self.crd)
        return self.uint_crd

    def Simulation_Caculate_Force(self, uint_crd, scaler, nl_atom_numbers, nl_atom_serial):
        '''simulation calculate force'''
        uint_crd = self.Simulation_Beforce_Caculate_Force()
        force = self.zero_frc
        if self.LJ_info_is_initialized:
            lj_force = self.lj_force_pme_direct_force(uint_crd, self.atom_LJ_type, self.charge, scaler, nl_atom_numbers,
                                                      nl_atom_serial, self.LJ_A, self.LJ_B)
            force = force + lj_force

        if self.pme_is_initialized:
            pme_excluded_force = self.pme_excluded_force(uint_crd, scaler, self.charge, self.excluded_list_start,
                                                         self.excluded_list, self.excluded_numbers)

            pme_reciprocal_force = self.pme_reciprocal_force(uint_crd, self.charge)
            force = force + pme_excluded_force + pme_reciprocal_force
        if self.nb14_is_initialized:
            nb14_force, _ = self.nb14_force_with_atom_energy(uint_crd, self.atom_LJ_type, self.charge,
                                                             scaler, self.nb14_atom_a, self.nb14_atom_b,
                                                             self.lj_scale_factor, self.cf_scale_factor,
                                                             self.LJ_A, self.LJ_B)
            force = force + nb14_force

        if self.bond_is_initialized:
            bond_force, _, _ = self.bond_force_with_atom_energy_virial(uint_crd, scaler, self.bond_atom_a,
                                                                       self.bond_atom_b, self.bond_k, self.bond_r0)
            force = force + bond_force
        if self.angle_is_initialized:
            angle_force, _ = self.angle_force_with_atom_energy(uint_crd, scaler, self.angle_atom_a,
                                                               self.angle_atom_b, self.angle_atom_c,
                                                               self.angle_k, self.angle_theta0)
            force = force + angle_force
        if self.dihedral_is_initialized:
            dihedral_force, _ = self.dihedral_force_with_atom_energy(uint_crd, scaler,
                                                                     self.dihedral_atom_a,
                                                                     self.dihedral_atom_b,
                                                                     self.dihedral_atom_c,
                                                                     self.dihedral_atom_d, self.ipn,
                                                                     self.pk, self.gamc, self.gams,
                                                                     self.pn)
            force = force + dihedral_force

        if self.restrain_is_initialized:
            _, _, restrain_frc = self.restrain_force_with_atom_energy_and_virial(self.restrain_list,
                                                                                 self.crd,
                                                                                 self.crd_ref,
                                                                                 self.box_length)
            force = force + restrain_frc

        return force

    def Simulation_Caculate_Energy(self, uint_crd, uint_dr_to_dr_cof):
        '''simulation calculate energy'''

        lj_energy = self.lj_energy(uint_crd, self.atom_LJ_type, self.charge, uint_dr_to_dr_cof, self.nl_atom_numbers,
                                   self.nl_atom_serial, self.LJ_A, self.LJ_B)

        lj_energy_sum = P.ReduceSum(True)(lj_energy)
        # lj_energy_sum = self.zero_fp_tensor

        reciprocal_energy, self_energy, direct_energy, correction_energy = self.pme_energy(uint_crd, self.charge,
                                                                                           self.nl_atom_numbers,
                                                                                           self.nl_atom_serial,
                                                                                           uint_dr_to_dr_cof,
                                                                                           self.excluded_list_start,
                                                                                           self.excluded_list,
                                                                                           self.excluded_numbers)
        ee_ene = reciprocal_energy + self_energy + direct_energy + correction_energy
        # ee_ene = self.zero_fp_tensor

        nb14_lj_energy = self.nb14_lj_energy(uint_crd, self.atom_LJ_type, self.charge, uint_dr_to_dr_cof,
                                             self.nb14_atom_a, self.nb14_atom_b, self.lj_scale_factor, self.LJ_A,
                                             self.LJ_B)
        nb14_cf_energy = self.nb14_cf_energy(uint_crd, self.atom_LJ_type, self.charge, uint_dr_to_dr_cof,
                                             self.nb14_atom_a, self.nb14_atom_b, self.cf_scale_factor)
        nb14_lj_energy_sum = P.ReduceSum(True)(nb14_lj_energy)
        nb14_cf_energy_sum = P.ReduceSum(True)(nb14_cf_energy)
        # nb14_lj_energy_sum = self.zero_fp_tensor
        # nb14_cf_energy_sum = self.zero_fp_tensor
        bond_energy = self.bond_energy(uint_crd, uint_dr_to_dr_cof, self.bond_atom_a, self.bond_atom_b, self.bond_k,
                                       self.bond_r0)
        bond_energy_sum = P.ReduceSum(True)(bond_energy)

        angle_energy = self.angle_energy(uint_crd, uint_dr_to_dr_cof, self.angle_atom_a, self.angle_atom_b,
                                         self.angle_atom_c, self.angle_k, self.angle_theta0)
        angle_energy_sum = P.ReduceSum(True)(angle_energy)

        dihedral_energy = self.dihedral_energy(uint_crd, uint_dr_to_dr_cof, self.dihedral_atom_a, self.dihedral_atom_b,
                                               self.dihedral_atom_c, self.dihedral_atom_d, self.ipn, self.pk, self.gamc,
                                               self.gams, self.pn)
        dihedral_energy_sum = P.ReduceSum(True)(dihedral_energy)

        total_energy = P.AddN()(
            [bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum,
             lj_energy_sum, ee_ene])
        return bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
               lj_energy_sum, ee_ene, total_energy

    def Simulation_Temperature(self):
        """calculate temperature"""
        res_ek_energy = self.mdtemp(self.res_start, self.res_end, self.velocity, self.mass)
        temperature = P.ReduceSum()(res_ek_energy)
        return temperature

    def Simulation_MDIterationLeapFrog_Liujian(self, inverse_mass, sqrt_mass_inverse, crd, frc, rand_state, random_frc):
        '''simulation leap frog iteration liujian'''
        if self.max_velocity <= 0:
            crd = self.md_iteration_leap_frog_liujian(inverse_mass, sqrt_mass_inverse, self.velocity, crd, frc,
                                                      self.acc,
                                                      rand_state, random_frc)
        else:
            crd = self.md_iteration_leap_frog_liujian_with_max_vel(inverse_mass, sqrt_mass_inverse, self.velocity, crd,
                                                                   frc, self.acc,
                                                                   rand_state, random_frc)
        vel = F.depend(self.velocity, crd)
        acc = F.depend(self.acc, crd)
        return vel, crd, acc

    def Simulation_MDIterationLeapFrog(self, force):
        '''simulation leap frog'''
        if self.max_velocity <= 0:
            res = self.md_iteration_leap_frog(self.velocity, self.crd, force, self.acc, self.mass_inverse)
        else:
            res = self.md_iteration_leap_frog_with_max_vel(self.velocity, self.crd, force, self.acc, self.mass_inverse)
        vel = F.depend(self.velocity, res)
        crd = F.depend(self.crd, res)
        return vel, crd, res

    def Simulation_MDInformationGradientDescent(self, force):
        # print("Simulation_MDInformationGradientDescent")
        res = self.md_information_gradient_descent(self.crd, force)
        self.velocity = self.zero_frc
        vel = F.depend(self.velocity, res)
        crd = F.depend(self.crd, res)
        return vel, crd, res

    def Main_Print(self, *args):
        """compute the temperature"""
        steps, temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene = list(args)
        if steps == 0:
            print("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                  "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_")

        temperature = temperature.asnumpy()
        total_potential_energy = total_potential_energy.asnumpy()
        print("{:>7.0f} {:>7.3f} {:>11.3f}".format(steps + 1, float(temperature), float(total_potential_energy)),
              end=" ")
        if self.bond.bond_numbers > 0:
            sigma_of_bond_ene = sigma_of_bond_ene.asnumpy()
            print("{:>10.3f}".format(float(sigma_of_bond_ene)), end=" ")
        if self.angle.angle_numbers > 0:
            sigma_of_angle_ene = sigma_of_angle_ene.asnumpy()
            print("{:>11.3f}".format(float(sigma_of_angle_ene)), end=" ")
        if self.dihedral.dihedral_numbers > 0:
            sigma_of_dihedral_ene = sigma_of_dihedral_ene.asnumpy()
            print("{:>14.3f}".format(float(sigma_of_dihedral_ene)), end=" ")
        if self.nb14.nb14_numbers > 0:
            nb14_lj_energy_sum = nb14_lj_energy_sum.asnumpy()
            nb14_cf_energy_sum = nb14_cf_energy_sum.asnumpy()
            print("{:>10.3f} {:>10.3f}".format(float(nb14_lj_energy_sum), float(nb14_cf_energy_sum)), end=" ")
        LJ_energy_sum = LJ_energy_sum.asnumpy()
        ee_ene = ee_ene.asnumpy()
        print("{:>7.3f}".format(float(LJ_energy_sum)), end=" ")
        print("{:>12.3f}".format(float(ee_ene)))
        if self.file is not None:
            self.file.write("{:>7.0f} {:>7.3f} {:>11.3f} {:>10.3f} {:>11.3f} {:>14.3f} {:>10.3f} {:>10.3f} {:>7.3f}"
                            " {:>12.3f}\n".format(steps, float(temperature), float(total_potential_energy),
                                                  float(sigma_of_bond_ene), float(sigma_of_angle_ene),
                                                  float(sigma_of_dihedral_ene), float(nb14_lj_energy_sum),
                                                  float(nb14_cf_energy_sum), float(LJ_energy_sum), float(ee_ene)))
        if self.datfile is not None:
            self.datfile.write(self.crd.asnumpy())

    def Main_Initial(self):
        """main initial"""
        if self.control.mdout:
            self.file = open(self.control.mdout, 'w')
            self.file.write("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                            "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_\n")
        if self.control.mdcrd:
            self.datfile = open(self.control.mdcrd, 'wb')

    def Main_Destroy(self):
        """main destroy"""
        if self.file is not None:
            self.file.close()
            print("Save .out file successfully!")
        if self.datfile is not None:
            self.datfile.close()
            print("Save .dat file successfully!")

    # 控压部分代码
    def Volume_Change_Attempt(self, boxlength, DeltaV_max):
        """Volume_Change_Attempt"""
        nrand = self.random((1, 1))
        DeltaV = nrand * DeltaV_max
        V = boxlength[0] * boxlength[1] * boxlength[2]
        # crd_scale_factor = Tensor(np.crbt((V + DeltaV) / V), mstype.float32)
        crd_scale_factor = self.pow((V + DeltaV) / V, -3)
        return crd_scale_factor

    def Update_Volume(self, factor):
        """Update_Volume"""
        self.CONSTANT_UINT_MAX_FLOAT = 4294967296.0
        # f_inv = 1.0 / factor
        self.box_length = factor * self.box_length
        self.crd_to_uint_crd_cof = self.CONSTANT_UINT_MAX_FLOAT / self.box_length
        self.quarter_crd_to_uint_crd_cof = 0.25 * self.crd_to_uint_crd_cof
        self.uint_dr_to_dr_cof = 1.0 / self.crd_to_uint_crd_cof
        self.uint_crd = self.crd_to_uint_crd_quarter(self.quarter_crd_to_uint_crd_cof, self.crd)

    def Neighbor_List_Update_Volume(self, box_length):
        """Neighbor_List_Update_Volume"""
        self.quarter_crd_to_uint_crd_cof = 0.25 * self.CONSTANT_UINT_MAX_FLOAT / box_length
        self.uint_dr_to_dr_cof = 1.0 / self.CONSTANT_UINT_MAX_FLOAT * box_length
        self.grid_length[0] = box_length[0] / self.Nx
        self.grid_length[1] = box_length[1] / self.Ny
        self.grid_length[2] = box_length[1] / self.Nz
        self.grid_length_inverse = 1.0 / self.grid_length

    def LJ_Update_Volume(self):
        """main destroy"""
        if self.LJ_info_is_initialized:
            # self.uint_dr_to_dr_cof = 1.0 / self.CONSTANT_UINT_MAX_FLOAT * self.box_length
            self.volume = self.box_length[0] * self.box_length[1] * self.box_length[2]

    def PME_Update_Volume(self, factor):
        """PME_Update_Volume"""
        factor_inverse = 1.0 / factor
        self.PME_inverse_box_vector[0] = self.fftx / self.box_length[0]
        self.PME_inverse_box_vector[1] = self.ffty / self.box_length[1]
        self.PME_inverse_box_vector[2] = self.fftz / self.box_length[2]
        self.PME_inverse_box_vector = factor_inverse * self.PME_inverse_box_vector
        self.beta = self.beta * factor
        # self.PME_BC = self.PME_BC * factor_inverse #scale list
        self.neutralizing_factor = self.pow(factor, 5.0)

    def Simple_Constrain_Update_Volume(self):
        """Simple_Constrain_Update_Volume"""
        if self.simple_constrain_is_initialized:
            self.quarter_crd_to_uint_crd_cof = 0.25 * self.CONSTANT_UINT_MAX_FLOAT / self.box_length
            self.uint_dr_to_dr_cof = 1.0 / self.CONSTANT_UINT_MAX_FLOAT * self.box_length
            self.volume = self.box_length[0] * self.box_length[1] * self.box_length[2]

    def Main_Volume_Change(self, factor):
        """Main_Volume_Change"""
        self.Update_Volume(factor)
        self.Neighbor_List_Update_Volume(self.box_length)
        _ = self.neighbor_list_update_nb(self.atom_numbers_in_grid_bucket, self.bucket,
                                         self.crd, self.box_length, self.grid_N,
                                         self.grid_length_inverse, self.atom_in_grid_serial,
                                         self.old_crd, self.crd_to_uint_crd_cof, self.uint_crd,
                                         self.pointer, self.nl_atom_numbers, self.nl_atom_serial,
                                         self.uint_dr_to_dr_cof, self.excluded_list_start, self.excluded_list,
                                         self.excluded_numbers, self.need_refresh_flag, self.refresh_count)  # Done
        self.LJ_Update_Volume()
        self.PME_Update_Volume(factor)
        self.Simple_Constrain_Update_Volume()
        # self.mol_map.Update_Volume(self.md_info.sys.box_length)

    def Main_Volume_Change_Largely(self):
        """Main_Volume_Change_Largely"""
        # re-initialize neighbor_list and pme
        _ = self.neighbor_list_update_forced_update(self.atom_numbers_in_grid_bucket, self.bucket,
                                                    self.crd, self.box_length, self.grid_N,
                                                    self.grid_length_inverse, self.atom_in_grid_serial,
                                                    self.old_crd, self.crd_to_uint_crd_cof, self.uint_crd,
                                                    self.pointer, self.nl_atom_numbers, self.nl_atom_serial,
                                                    self.uint_dr_to_dr_cof, self.excluded_list_start,
                                                    self.excluded_list,
                                                    self.excluded_numbers, self.need_refresh_flag,
                                                    self.refresh_count)

    def Check_MC_Barostat_Accept(self):
        """Check_MC_Barostat_Accept"""
        self.total_count = self.total_count + 1
        rand_num = self.random((1, 1))
        if rand_num[0] < self.mc_baro_accept_possibility:
            self.reject = 0
            self.accept_count += 1
        else:
            self.reject = 1
        return self.reject

    def Delta_V_Max_Update(self):
        """Delta_V_Max_Update"""
        if self.total_count % self.check_interval == 0:
            self.accept_rate = 100.0 * self.accept_count / self.total_count
            if self.accept_rate < self.accept_rate_low:
                self.total_count = 0
                self.accept_count = 0
                self.DeltaV_max = self.DeltaV_max * 0.9
            if self.accept_rate > self.accept_rate_high:
                self.total_count = 0
                self.accept_count = 0
                self.DeltaV_max = self.DeltaV_max * 1.1

    def Main_iteration_presssure(self, steps, force):
        """Main_iteration_presssure"""
        if self.mc_baro_is_initialized and steps % self.mc_baro.update_interval == 0:
            # old energy
            self.mc_baro_energy_old = self.potential
            self.frc_backup = self.frc
            self.crd_backup = self.crd
            self.Volume_Change_Attempt(self.box_length, 200)

            # change coordinates
            if self.is_molecule_map_output:
                nowrap_crd = self.Calculate_No_Wrap_Crd()
                self.crd, _ = self.Residue_Crd_Map(nowrap_crd)
                _ = self.refresh_boxmaptimes(self.crd, self.old_crd, 1.0 / self.box_length, self.box_map_times)
            else:
                self.crd = self.crd * self.crd_scale_factor  # scale list

            # change volume
            self.Main_Volume_Change(self.crd_scale_factor)
            self.system_reinitializing_count += 1

            # new energy
            _ = self.Simulation_Caculate_Force(self.uint_crd, self.uint_dr_to_dr_cof, self.nl_atom_numbers,
                                               self.nl_atom_serial)

            self.energy_new = self.potential

            # calculate accepted rate
            if self.scale_coordinate_by_residue:
                self.extra_term = self.target_pressure * self.DeltaV - \
                                  self.residue_numbers * self.CONSTANT_kB * \
                                  self.target_temperature * self.log(self.VDevided)
            else:
                self.extra_term = self.target_pressure * self.DeltaV - \
                                  self.atom_numbers * self.CONSTANT_kB * \
                                  self.target_temperature * self.log(self.VDevided)

            self.mc_baro_accept_possibility = self.mc_baro_energy_new - self.mc_baro_energy_old + self.extra_term
            self.mc_baro.mc_baro_accept_possibility = self.exp(
                -self.mc_baro_accept_possibility / (self.CONSTANT_kB * self.target_temperature))

            # check if accepted
            if self.Check_MC_Barostat_Accept():
                # if accept, refresh
                self.crd_scale_factor = 1.0 / self.crd_scale_factor
                self.crd = self.crd_backup
                self.Main_Volume_Change(self.crd_scale_factor)
                self.system_reinitializing_count += 1
                _ = self.neighbor_list_update_mc(self.atom_numbers_in_grid_bucket, self.bucket,
                                                 self.crd, self.box_length, self.grid_N,
                                                 self.grid_length_inverse, self.atom_in_grid_serial,
                                                 self.old_crd, self.crd_to_uint_crd_cof, self.uint_crd,
                                                 self.pointer, self.nl_atom_numbers, self.nl_atom_serial,
                                                 self.uint_dr_to_dr_cof, self.excluded_list_start, self.excluded_list,
                                                 self.excluded_numbers, self.need_refresh_flag,
                                                 self.refresh_count)
                self.frc = force
                self.frc = self.frc_backup

            # reinitialized
            if self.system_reinitializing_count >= 20000 or (not self.reject and (
                    self.mc_baro_newV > 1.331 * self.mc_baro_V0 or self.mc_baro_newV < 0.729 * self.mc_baro.V0)):
                self.Main_Volume_Change_Largely()
                self.mc_baro_V0 = self.mc_baro_newV
                self.system_reinitializing_count = self.zero_fp_tensor
            self.Delta_V_Max_Update()

    def Constrain(self):
        """Constrain"""
        constrain_frc = self.zero_frc
        for _ in range(self.iteration_numbers):
            test_uint_crd = self.refresh_uint_crd(self.crd, self.quarter_crd_to_uint_crd_cof, constrain_frc,
                                                  self.mass_inverse)
            if self.need_pressure:
                force, _ = self.constrain_force_cycle_with_virial(test_uint_crd, self.uint_dr_to_dr_cof,
                                                                  self.last_pair_dr, self.atom_i_serials,
                                                                  self.atom_j_serials, self.constant_rs,
                                                                  self.constrain_ks)
            else:
                force = self.constrain_force_cycle(test_uint_crd, self.uint_dr_to_dr_cof, self.last_pair_dr,
                                                   self.atom_i_serials,
                                                   self.atom_j_serials, self.constant_rs, self.constrain_ks)
            constrain_frc = constrain_frc + force

        res = self.refresh_crd_vel(self.crd, self.velocity, constrain_frc, self.mass_inverse)
        crd = self.depend(self.crd, res)
        vel = self.depend(self.velocity, res)

        return crd, vel, res

    def Main_Iteration(self, steps, force):
        '''Main_Iteration'''
        # self.Main_iteration_presssure(steps, force)
        # Remember_Last_Coordinates
        # pressure control 1
        if self.simple_constrain_is_initialized:
            self.last_pair_dr = self.last_crd_to_dr(self.crd, self.quarter_crd_to_uint_crd_cof, self.uint_dr_to_dr_cof,
                                                    self.atom_i_serials,
                                                    self.atom_j_serials, self.constant_rs, self.constrain_ks)

        if self.mode == 0:  # NVE
            self.velocity, self.crd, _ = self.Simulation_MDIterationLeapFrog(force)
        elif self.mode == -1:  # Minimization
            _ = self.Simulation_MDInformationGradientDescent(force)
        else:
            if self.liujian_info_is_initialized:
                self.velocity, self.crd, _ = self.Simulation_MDIterationLeapFrog_Liujian(self.mass_inverse,
                                                                                         self.sqrt_mass, self.crd,
                                                                                         force,
                                                                                         self.rand_state,
                                                                                         self.random_force)

        if self.simple_constrain_is_initialized:
            self.crd, self.velocity, res1 = self.Constrain()
        else:
            res1 = self.zero_fp_tensor

        # MD_Information_Crd_To_Uint_Crd
        self.uint_crd = self.crd_to_uint_crd_quarter(self.quarter_crd_to_uint_crd_cof, self.crd)
        res2 = self.neighbor_list_update(self.atom_numbers_in_grid_bucket,
                                         self.bucket,
                                         self.crd,
                                         self.box_length,
                                         self.grid_N,
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

        res3 = self.refresh_boxmaptimes(self.crd, self.old_crd, 1.0 / self.box_length, self.box_map_times)

        return self.velocity, self.crd, res1, res2, res3

    def Calculate_No_Wrap_Crd(self):
        """Calculate_No_Wrap_Crd"""
        nowrap_crd = self.box_map_times * self.box_length + self.crd
        return nowrap_crd

    def Residue_Crd_Map(self, nowrap_crd):
        """Residue_Crd_Map"""
        center_of_mass = self.getcenterofmass(self.res_start, self.res_end, nowrap_crd, self.mass,
                                              self.res_mass_inverse)

        res = self.mapcenterofmass(self.res_start, self.res_end, center_of_mass, self.box_length, nowrap_crd, self.crd)

        return self.crd, res

    def construct(self, step, print_step):
        '''construct'''
        # self.last_crd = self.crd
        if step == 0:
            res = self.neighbor_list_update_forced_update(self.atom_numbers_in_grid_bucket,
                                                          self.bucket,
                                                          self.crd,
                                                          self.box_length,
                                                          self.grid_N,
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
        else:
            res = self.zero_fp_tensor
        force = self.Simulation_Caculate_Force(self.uint_crd, self.uint_dr_to_dr_cof, self.nl_atom_numbers,
                                               self.nl_atom_serial)
        if step == 0:
            self.rand_state = self.setup_random_state()

        self.velocity, self.crd, res1, res2, res3 = self.Main_Iteration(step + 1, force)
        temperature = self.Simulation_Temperature()
        if print_step == 0:
            bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
            lj_energy_sum, ee_ene, total_energy = self.Simulation_Caculate_Energy(self.uint_crd, self.uint_dr_to_dr_cof)
        else:
            bond_energy_sum = self.zero_fp_tensor
            angle_energy_sum = self.zero_fp_tensor
            dihedral_energy_sum = self.zero_fp_tensor
            nb14_lj_energy_sum = self.zero_fp_tensor
            nb14_cf_energy_sum = self.zero_fp_tensor
            lj_energy_sum = self.zero_fp_tensor
            ee_ene = self.zero_fp_tensor
            total_energy = self.zero_fp_tensor
        return temperature, total_energy, bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, \
               nb14_cf_energy_sum, lj_energy_sum, ee_ene, res, res1, res2, res3
