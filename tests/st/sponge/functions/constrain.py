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
'''constrain'''

import mindspore.ops as ops
import mindspore.numpy as np
from mindsponge.md.functions.common import get_periodic_displacement

int32_four = np.array(4, dtype=np.int32)


def refresh_uint_crd(half_exp_gamma_plus_half, crd, quarter_cof, frc,
                     mass_inverse):
    '''refresh uint crd '''
    mass_inverse_mul = np.expand_dims(mass_inverse, -1)
    crd_lin = (crd + half_exp_gamma_plus_half * frc * mass_inverse_mul).astype("float32")
    tempi = (crd_lin * quarter_cof).astype("int32")
    uint_crd = (tempi * int32_four).astype("uint32")

    return uint_crd


def constrain_force_cycle_with_virial(uint_crd, scaler, pair_dr, atom_i_serials,
                                      atom_j_serials, constant_rs, constrain_ks, frc, virial):
    '''constrain force cycle with virial'''
    dr = get_periodic_displacement(uint_crd[atom_i_serials], uint_crd[atom_j_serials], scaler)
    r_1 = 1. / np.norm(dr, axis=-1)
    frc_abs = (1. - constant_rs * r_1) * constrain_ks
    frc_lin = np.expand_dims(frc_abs, -1) * pair_dr
    frc_sum = (frc_lin * pair_dr).astype("float32")

    virial -= np.sum(frc_sum, -1)

    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_j_serials, -1), frc_lin)
    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_i_serials, -1), -frc_lin)

    return frc, virial


def constrain_force_cycle(uint_crd, scaler, pair_dr, atom_i_serials,
                          atom_j_serials, constant_rs, constrain_ks, frc):
    '''constrain force cycle'''
    dr = get_periodic_displacement(uint_crd[atom_i_serials], uint_crd[atom_j_serials], scaler)
    r_1 = 1. / np.norm(dr, axis=-1)
    frc_abs = (1. - constant_rs * r_1) * constrain_ks
    frc_lin = np.expand_dims(frc_abs, -1) * pair_dr

    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_j_serials, -1), frc_lin)
    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_i_serials, -1), -frc_lin)

    return frc


def constrain(atom_numbers, constrain_pair_numbers, iteration_numbers, half_exp_gamma_plus_half,
              crd, quarter_cof, mass_inverse, scalar, pair_dr, atom_i_serials, atom_j_serials, constant_rs,
              constrain_ks,
              need_pressure):
    """
    Calculate the constraint force and virial depends on pressure calculation.

    Args:
        atom_numbers (int): the number of atoms N.
        constrain_pair_numbers (int): the number of constrain pairs M.
        iteration_numbers (int): the number of iteration numbers p.
        half_exp_gamma_plus_half (float32): half exp_gamma plus half q.
        crd(Tensor, float32): [N, 3], the coordinate of each atom.
        quarter_cof (Tensor, float32): [3,], the coefficient mapping coordinates to uint coordinates.
        mass_inverse(Tensor, float32): [N, ], the inverse value of mass of each atom.
        scalar (Tensor, float32): [3,], the 3-D scale factor (x, y, z),
        pair_dr(Tensor, float32): [M, 3], the displacement vector of each constrained atom pair.
        atom_i_serials(Tensor, int32): [M, ], the first atom index of each constrained atom pair.
        atom_j_serials(Tensor, int32): [M, ], the second atom index of each constrained atom pair.
        constant_rs(Tensor, float32): [M, ], the constrained distance of each constrained atom pair.
        constrain_ks(Tensor, float32): [M, ], the constrained distance of each constrained atom pair.
        need_pressure(Tensor, int32), [1, ]or [], if need pressure, 1 else 0.

    Outputs:
        uint_crd(Tensor, float32): [N, 3], the unsigned int coordinate value of each atom.
        frc(Tensor, float32): [N, 3], the constraint force on each atom.
        virial(Tensor, float32): [M, ], the constraint virial on each atom

    """
    frc = np.zeros(crd.shape, np.float32)
    virial = np.zeros((constrain_pair_numbers,), np.float32)
    uint_crd = np.zeros((atom_numbers, 3)).astype("uint32")

    while iteration_numbers > 0:
        uint_crd = refresh_uint_crd(half_exp_gamma_plus_half, crd, quarter_cof, frc,
                                    mass_inverse)
        if need_pressure:
            frc, virial = constrain_force_cycle_with_virial(uint_crd, scalar,
                                                            pair_dr,
                                                            atom_i_serials,
                                                            atom_j_serials, constant_rs, constrain_ks, frc, virial)
        else:
            frc = constrain_force_cycle(uint_crd, scalar, pair_dr, atom_i_serials,
                                        atom_j_serials, constant_rs, constrain_ks, frc)
        iteration_numbers -= 1

    return uint_crd, frc, virial
