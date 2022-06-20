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
'''bond force with atom energy with virial'''

import mindspore.numpy as np
import mindspore.ops as ops
from .common import get_periodic_displacement


def bond_force_with_atom_energy_with_virial(atom_numbers, uint_crd_f, scalar_f, atom_a, atom_b, bond_k,
                                            bond_r0):
    """
    Calculate bond force and harmonic potential energy together.

    The calculation formula is the same as operator BondForce() and BondEnergy().

    Args:
        atom_numbers (int): the number of atoms N.
        uint_crd_f (Tensor, uint32 ) - [N, 3], the unsigned int coordinate value of each atom.
        scalar_f (Tensor, float32): [3,], the 3-D scale factor (x, y, z),
            between the real space float coordinates and the unsigned int coordinates.
        atom_a (Tensor, int32): [M,], the first atom index of each bond.
        atom_b (Tensor, int32): [M,], the second atom index of each bond.
        bond_k (Tensor, float32): [M,], the force constant of each bond.
        bond_r0 (Tensor, float32): [M,], the equlibrium length of each bond.

    Outputs:
        frc_f (Tensor, float32): [N, 3], same as operator BondForce().
        atom_energy (Tensor, float32): [N,], same as atom_ene in operator BondAtomEnergy().
        atom_virial (Tensor, float32): [N,], the virial of each atom

    Supported Platforms:
        ``GPU``
    """
    frc = np.zeros(uint_crd_f.shape, np.float32)
    atom_energy = np.zeros(atom_numbers)
    atom_virial = np.zeros(atom_numbers)

    dr = get_periodic_displacement(uint_crd_f[atom_a], uint_crd_f[atom_b], scalar_f)
    abs_r = np.norm(dr, axis=-1)
    r_1 = 1. / abs_r
    tempf = abs_r - bond_r0
    f = 2 * np.expand_dims(tempf * r_1 * bond_k, -1) * dr

    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_a, -1), -f)
    frc = ops.tensor_scatter_add(frc, np.expand_dims(atom_b, -1), f)

    atom_energy_val = bond_k * tempf * tempf
    atom_virial_val = -2 * tempf * bond_k * abs_r
    atom_energy = ops.tensor_scatter_add(atom_energy, np.expand_dims(atom_a, -1), atom_energy_val)
    atom_virial = ops.tensor_scatter_add(atom_virial, np.expand_dims(atom_a, -1), atom_virial_val)
    return frc, atom_energy, atom_virial
