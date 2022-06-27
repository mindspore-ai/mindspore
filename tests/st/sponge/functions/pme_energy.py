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
'''pme energy'''
import mindspore.numpy as mnp
from .common import PI, get_zero_tensor, get_full_tensor
from .pme_common import scale_list, pme_a_near, pme_q_spread, pme_direct_energy, pme_energy_reciprocal, \
    pme_excluded_energy_correction_sparse, pme_excluded_energy_correction_dense, pme_energy_product,\
    get_pme_kxyz, PERIODIC_FACTOR_INVERSE, fft3d

CUT_OFF = 10.0


def pme_energy(atom_numbers, beta, fftx, ffty, fftz, pme_bc, uint_crd, charge,
               nl_numbers, nl_serial, scaler, **kargs):
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
        Keyword Arguments:
        excluded_matrix (Tensor, int32): [N, k] containing the excluded atoms for each atom, where k
          is the maximum number of excluded atoms for all atoms.
        excluded_list (Tensor, int32):[E,] containing the CSR format column indices of excluded atom.
          E is the number of excluded atoms.
        atom_excluded_index (Tensor, int32):[E,] containing the row indices corresponding excluded_list,
        i.e. each CSR format row index is replicated to map to one CSR format column index.

    Outputs:
        reciprocal_ene  (float) - the reciprocal term of PME energy.
        self_ene (float) - the self term of PME energy.
        direct_ene (float) - the direct term of PME energy.
        correction_ene (float) - the correction term of PME energy.

    Supported Platforms:
        ``GPU``
    """
    pme_nin = ffty * fftz
    pme_nall = fftx * ffty * fftz

    pme_kxyz = get_pme_kxyz() # (64, 3)

    pme_uxyz = get_full_tensor((atom_numbers, 3), 2 ** 30, mnp.uint32)
    pme_atom_near = get_zero_tensor((atom_numbers, 64), mnp.int32)
    pme_frxyz, pme_uxyz, pme_atom_near = pme_a_near(uint_crd, pme_atom_near, pme_nin,
                                                    PERIODIC_FACTOR_INVERSE * fftx,
                                                    PERIODIC_FACTOR_INVERSE * ffty,
                                                    PERIODIC_FACTOR_INVERSE * fftz, atom_numbers,
                                                    fftx, ffty, fftz, pme_kxyz, pme_uxyz)

    pme_q = get_full_tensor(pme_nall, 0, mnp.float32)
    pme_q = pme_q_spread(pme_atom_near, charge, pme_frxyz, pme_q, pme_kxyz, atom_numbers)

    pme_q = pme_q.reshape(fftx, ffty, fftz).astype('float32')
    pme_fq = fft3d(pme_q)

    reciprocal_ene = pme_energy_reciprocal(pme_fq, pme_bc.reshape((fftx, ffty, fftz // 2 + 1)))

    self_ene = pme_energy_product(charge, charge)
    self_ene = scale_list(1, self_ene, -beta / mnp.sqrt(PI))

    direct_ene = pme_direct_energy(atom_numbers, nl_numbers, nl_serial, uint_crd, scaler, charge, beta,
                                   CUT_OFF * CUT_OFF)

    if "excluded_matrix" in kargs.keys():
        correction_ene = pme_excluded_energy_correction_dense(uint_crd, scaler, charge, beta, kargs["excluded_matrix"])
    else:
        correction_ene = pme_excluded_energy_correction_sparse(uint_crd, scaler, charge, beta,
                                                               kargs["excluded_list"], kargs["atom_excluded_index"])
    res = (mnp.atleast_1d(reciprocal_ene), mnp.atleast_1d(self_ene), \
           mnp.atleast_1d(direct_ene), mnp.atleast_1d(correction_ene))
    return res
