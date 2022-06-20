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
'''dihedral energy'''
from mindspore import numpy as np

from .common import get_periodic_displacement


def dihedral_energy(dihedral_numbers, uint_crd_f, scalar_f, atom_a, atom_b, atom_c, atom_d, ipn, pk, gamc, gams, pn):
    """
    Calculate the potential energy caused by dihedral terms for each 4-atom pair.
    Assume our system has N atoms and M dihedral terms.

    .. math::

        E = k(1 + cos(n*phi - phi_0))

    Args:
        dihedral_numbers (int) - the number of dihedral terms M.
        uint_crd_f (Tensor, uint32): [N, 3], the unsigned int coordinates
            value of each atom.
        scalar_f (Tensor, float32): [3,], the 3-D scale factor between
            the real space float coordinates and the unsigned int coordinates.
        atom_a (Tensor, int32): [M,], the 1st atom index of each dihedral.
        atom_b (Tensor, int32): [M,], the 2nd atom index of each dihedral.
        atom_c (Tensor, int32): [M,], the 3rd atom index of each dihedral.
        atom_d (Tensor, int32): [M,], the 4th atom index of each dihedral.
            4 atoms are connected in the form a-b-c-d.
        ipn (Tensor, int32): [M,], the period of dihedral angle of each dihedral.
        pk (Tensor, float32): [M,], the force constant of each dihedral.
        gamc (Tensor, float32): [M,], k*cos(phi_0) of each dihedral.
        gams (Tensor, float32): [M,], k*sin(phi_0) of each dihedral.
        pn (Tensor, float32): [M,], the floating point form of ipn.

    Outputs:
        ene (Tensor, float32): [M,], the potential energy for each
          dihedral term.

    Supported Platforms:
        ``GPU``
    """
    drij = get_periodic_displacement(uint_crd_f[atom_a], uint_crd_f[atom_b], scalar_f)
    drkj = get_periodic_displacement(uint_crd_f[atom_c], uint_crd_f[atom_b], scalar_f)
    drkl = get_periodic_displacement(uint_crd_f[atom_c], uint_crd_f[atom_d], scalar_f)

    r1 = np.cross(drij, drkj)
    r2 = np.cross(drkl, drkj)

    r1_1 = 1. / np.norm(r1, axis=-1)
    r2_1 = 1. / np.norm(r2, axis=-1)
    r1_1_r2_1 = r1_1 * r2_1

    phi = np.sum(r1 * r2, -1) * r1_1_r2_1
    phi = np.clip(phi, -0.999999, 0.999999)
    phi = np.arccos(phi)

    sign = np.sum(np.cross(r2, r1) * drkj, -1)
    phi = np.copysign(phi, sign)

    phi = np.pi - phi
    nphi = pn * phi

    cos_nphi = np.cos(nphi)
    sin_nphi = np.sin(nphi)

    return pk + cos_nphi * gamc + sin_nphi * gams
