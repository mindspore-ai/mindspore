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
'''pme energy update'''
import mindspore.numpy as mnp
from .pme_energy import pme_energy


def pme_energy_update(atom_numbers, beta, fftx, ffty, fftz, pme_bc, uint_crd, charge,
                      nl_numbers, nl_serial, scaler, excluded_list, atom_excluded_index, neutralizing_factor):
    """
    Calculate the Coulumb energy of the system using PME method.

    .. math::

        E = sum_{ij} q_iq_j/r_{ij}

    Args:
        atom_numbers (int): the number of atoms, N.
        beta (float): the PME beta parameter, determined by the
                       non-bond cutoff value and simulation precision tolerance.
        fftx (int): the number of points for Fourier transform in dimension X.
        ffty (int): the number of points for Fourier transform in dimension Y.
        fftz (int): the number of points for Fourier transform in dimension Z.
        uint_crd (Tensor, uint32): [N, 3], the unsigned int coordinates value of each atom.
        charge (Tensor, float32): [N,], the charge carried by each atom.
        nl_numbers - (Tensor, int32): [N,], the each atom.
        nl_serial - (Tensor, int32): [N, 800], the neighbor list of each atom, the max number is 800.
        scaler (Tensor, float32): [3,], the scale factor between real space
          coordinates and its unsigned int value.
        excluded_list (Tensor, int32):[E,] containing the CSR format column indices of excluded atom.
          E is the number of excluded atoms.
        atom_excluded_index (Tensor, int32):[E,] containing the row indices corresponding excluded_list,
        i.e. each CSR format row index is replicated to map to one CSR format column index.
        neutralizing_factor -(Tensor, float32): [1, ] the factor parameter to be updated in pressure calculation.

    Outputs:
        reciprocal_ene  (float) - the reciprocal term of PME energy.
        self_ene (float) - the self term of PME energy.
        direct_ene (float) - the direct term of PME energy.
        correction_ene (float) - the correction term of PME energy.

    Supported Platforms:
        ``GPU``
    """
    reciprocal_ene, self_ene, direct_ene, correction_ene = pme_energy(atom_numbers, beta, fftx, ffty, fftz,
                                                                      pme_bc, uint_crd, charge, nl_numbers,
                                                                      nl_serial, scaler, excluded_list=excluded_list,
                                                                      atom_excluded_index=atom_excluded_index)

    d_beta = mnp.sum(charge, -1)
    self_ene += neutralizing_factor * d_beta * d_beta
    res = (mnp.atleast_1d(reciprocal_ene), mnp.atleast_1d(self_ene), \
           mnp.atleast_1d(direct_ene), mnp.atleast_1d(correction_ene))
    return res
