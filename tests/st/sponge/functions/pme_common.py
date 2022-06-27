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
'''pme common'''
import mindspore.numpy as mnp
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import constexpr

from .common import get_neighbour_index, get_periodic_displacement

PERIODIC_FACTOR_INVERSE = 2.32830643e-10
pme_ma = mnp.array([1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0])
pme_mb = mnp.array([0, 0.5, -1, 0.5])
pme_mc = mnp.array([0, 0.5, 0, -0.5])
pme_md = mnp.array([0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0])
fft3d = ops.FFT3D()
ifft3d = ops.IFFT3D()
real = ops.Real()
conj = ops.Conj()


@constexpr
def to_tensor(args):
    return mnp.array(args)


def scale_list(element_numbers, tensor, scaler):
    """Scale values in `tensor`."""
    if tensor.ndim > 0 and len(tensor) > element_numbers:
        tensor = tensor[:element_numbers]
    return tensor * scaler


# pylint: disable=too-many-function-args
def pme_a_near(uint_crd, pme_atom_near, pme_nin, periodic_factor_inverse_x,
               periodic_factor_inverse_y, periodic_factor_inverse_z, atom_numbers,
               fftx, ffty, fftz, pme_kxyz, pme_uxyz):
    '''pme atom near'''
    periodic_factor_inverse_xyz = to_tensor(
        (periodic_factor_inverse_x, periodic_factor_inverse_y, periodic_factor_inverse_z))
    tempf = uint_crd.astype('float32') * periodic_factor_inverse_xyz
    tempu = tempf.astype('int32')
    tempu = ops.depend(tempu, tempu)
    pme_frxyz = tempf - tempu

    cond = mnp.not_equal(pme_uxyz.astype(mnp.int32), tempu).any(1, True)
    pme_uxyz = mnp.where(cond, tempu, pme_uxyz)

    tempu = tempu.reshape(atom_numbers, 1, 3)
    kxyz = tempu - pme_kxyz.astype(mnp.int32)
    kxyz_plus = kxyz + mnp.array([fftx, ffty, fftz])
    kxyz = ops.select(kxyz < 0, kxyz_plus, kxyz)

    kxyz = kxyz * to_tensor((pme_nin, fftz, 1)).reshape(1, 1, 3)
    temp_near = mnp.sum(kxyz.astype(mnp.float32), -1).astype(mnp.int32)
    pme_atom_near = mnp.where(cond, temp_near, pme_atom_near)

    return pme_frxyz, pme_uxyz, pme_atom_near


def pme_q_spread(pme_atom_near, charge, pme_frxyz, pme_q, pme_kxyz, atom_numbers):
    '''pme q spread'''
    pme_kxyz = pme_kxyz.astype(mnp.int32)
    pme_ma_new = pme_ma[pme_kxyz]
    pme_mb_new = pme_mb[pme_kxyz]
    pme_mc_new = pme_mc[pme_kxyz]
    pme_md_new = pme_md[pme_kxyz]

    tempf = pme_frxyz.reshape(atom_numbers, 1, 3) # (N, 1, 3)
    tempf2 = tempf * tempf # (N, 1, 3)
    temp_charge = charge.reshape(atom_numbers, 1) # (N, 1)

    tempf = pme_ma_new * tempf * tempf2 + pme_mb_new * tempf2 + pme_mc_new * tempf + pme_md_new # (N, 64, 3)

    tempq = temp_charge * tempf[..., 0] * tempf[..., 1] * tempf[..., 2] # (N, 64)
    index = pme_atom_near.ravel() # (N * 64,)
    tempq = tempq.ravel() # (N * 64,)
    pme_q = ops.tensor_scatter_add(pme_q, mnp.expand_dims(index, -1), tempq)

    return pme_q


# pylint: disable=too-many-arguments
def pme_direct_energy(atom_numbers, nl_numbers, nl_serial, uint_crd, boxlength, charge, beta, cutoff_square):
    '''pme direct energy'''
    r2 = uint_crd[nl_serial]

    dr_xyz = get_periodic_displacement(r2, mnp.expand_dims(uint_crd, 1), boxlength)
    dr2 = mnp.sum(dr_xyz * dr_xyz, -1)

    dr_abs = mnp.sqrt(dr2)
    charge_i = charge.reshape(-1, 1)
    charge_j = charge[nl_serial]

    ene_temp = charge_i * charge_j * ops.erfc(beta * dr_abs)
    where_zeros = dr_abs == 0.
    dr_abs[where_zeros] = 1.
    ene_temp = ene_temp / dr_abs

    idx = get_neighbour_index(atom_numbers, nl_serial.shape[1])
    mask = mnp.logical_and(dr2 < cutoff_square, idx < mnp.expand_dims(nl_numbers, -1))

    ene_lin = mnp.sum(ene_temp * mask)
    return ene_lin


@constexpr
def get_pme_kxyz():
    k = mnp.arange(4)
    x = mnp.repeat(k, 16).reshape(64, 1)
    y = F.tile(mnp.repeat(k, 4), (4, 1)).reshape(64, 1)
    z = F.tile(k, (16, 1)).reshape(64, 1)
    pme_kxyz = mnp.column_stack((x, y, z)).astype('uint32')
    return pme_kxyz


def pme_energy_reciprocal(pme_fq, bc):
    return mnp.sum(real(conj(pme_fq) * pme_fq) * bc)


def pme_energy_product(tensor1, tensor2):
    return mnp.sum(tensor1 * tensor2)


def pme_excluded_energy_correction_dense(uint_crd, scaler, charge, pme_beta, excluded_matrix):
    '''pme excluded energy correction'''
    mask = (excluded_matrix > -1)
    excluded_crd = uint_crd[excluded_matrix]

    dr_xyz = get_periodic_displacement(excluded_crd, mnp.expand_dims(uint_crd, 1), scaler)
    dr2 = mnp.sum(dr_xyz * dr_xyz, -1)
    dr_abs = mnp.sqrt(dr2)
    beta_dr = pme_beta * dr_abs

    excluded_charge = charge[excluded_matrix]
    charge_mul = mnp.expand_dims(charge, 1) * excluded_charge

    ene_lin = charge_mul * ops.erf(beta_dr)
    where_zeros = dr_abs == 0.
    dr_abs[where_zeros] = 1.
    ene_lin = ene_lin / dr_abs * mask

    return 0. - mnp.sum(ene_lin * mask)


def pme_excluded_energy_correction_sparse(uint_crd, scaler, charge, pme_beta, excluded_list, atom_excluded_index):
    '''pme excluded energy correction'''
    charge_mul = charge[atom_excluded_index] * charge[excluded_list]

    dr = get_periodic_displacement(uint_crd[excluded_list], uint_crd[atom_excluded_index], scaler)
    dr_2 = mnp.sum(mnp.square(dr), -1)

    dr_abs = mnp.sqrt(dr_2)
    beta_dr = pme_beta * dr_abs

    ene_lin = charge_mul * ops.erf(beta_dr) / dr_abs
    ene = mnp.sum(ene_lin)

    return 0. - ene
