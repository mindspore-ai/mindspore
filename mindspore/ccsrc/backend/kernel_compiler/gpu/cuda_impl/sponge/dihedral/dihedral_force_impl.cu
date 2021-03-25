/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/dihedral/dihedral_force_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void DihedralForceKernel(int dihedral_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *scaler,
                                    const int *atom_a, const int *atom_b, const int *atom_c, const int *atom_d,
                                    const int *ipn, const float *pk, const float *gamc, const float *gams,
                                    const float *pn, VECTOR *frc) {
  int dihedral_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (dihedral_i < dihedral_numbers) {
    int atom_i = atom_a[dihedral_i];
    int atom_j = atom_b[dihedral_i];
    int atom_k = atom_c[dihedral_i];
    int atom_l = atom_d[dihedral_i];

    int temp_ipn = ipn[dihedral_i];

    float temp_pn = pn[dihedral_i];
    float temp_gamc = gamc[dihedral_i];
    float temp_gams = gams[dihedral_i];

    VECTOR drij = Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler[0]);
    VECTOR drkj = Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler[0]);
    VECTOR drkl = Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_l], scaler[0]);

    VECTOR r1 = drij ^ drkj;
    VECTOR r2 = drkl ^ drkj;

    float r1_1 = rnorm3df(r1.x, r1.y, r1.z);
    float r2_1 = rnorm3df(r2.x, r2.y, r2.z);
    float r1_2 = r1_1 * r1_1;
    float r2_2 = r2_1 * r2_1;
    float r1_1_r2_1 = r1_1 * r2_1;

    float phi = r1 * r2 * r1_1_r2_1;
    phi = fmaxf(-0.999999, fminf(phi, 0.999999));
    phi = acosf(phi);

    float sign = (r2 ^ r1) * drkj;
    copysignf(phi, sign);

    phi = CONSTANT_Pi - phi;

    float nphi = temp_pn * phi;

    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);
    float cos_nphi = cosf(nphi);
    float sin_nphi = sinf(nphi);

    float dE_dphi;
    if (fabsf(sin_phi) < 1e-6) {
      temp_ipn *= temp_ipn % 2;  //  (((temp_ipn - 1) & 1) ^ 1)
      dE_dphi = temp_gamc * (temp_pn - temp_ipn + temp_ipn * cos_phi);
    } else {
      dE_dphi = temp_pn * (temp_gamc * sin_nphi - temp_gams * cos_nphi) / sin_phi;
    }

    VECTOR dphi_dr1 = r1_1_r2_1 * r2 + cos_phi * r1_2 * r1;
    VECTOR dphi_dr2 = r1_1_r2_1 * r1 + cos_phi * r2_2 * r2;

    VECTOR dE_dri = dE_dphi * drkj ^ dphi_dr1;
    VECTOR dE_drl = dE_dphi * dphi_dr2 ^ drkj;
    VECTOR dE_drj_part = dE_dphi * ((drij ^ dphi_dr1) + (drkl ^ dphi_dr2));

    VECTOR fi = dE_dri;
    VECTOR fj = dE_drj_part - dE_dri;
    VECTOR fk = -dE_drl - dE_drj_part;
    VECTOR fl = dE_drl;

    atomicAdd(&frc[atom_i].x, fi.x);
    atomicAdd(&frc[atom_i].y, fi.y);
    atomicAdd(&frc[atom_i].z, fi.z);
    atomicAdd(&frc[atom_j].x, fj.x);
    atomicAdd(&frc[atom_j].y, fj.y);
    atomicAdd(&frc[atom_j].z, fj.z);
    atomicAdd(&frc[atom_k].x, fk.x);
    atomicAdd(&frc[atom_k].y, fk.y);
    atomicAdd(&frc[atom_k].z, fk.z);
    atomicAdd(&frc[atom_l].x, fl.x);
    atomicAdd(&frc[atom_l].y, fl.y);
    atomicAdd(&frc[atom_l].z, fl.z);
  }
}

void DihedralForce(int dihedral_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f,
                   const int *atom_a, const int *atom_b, const int *atom_c, const int *atom_d, const int *ipn,
                   const float *pk, const float *gamc, const float *gams, const float *pn, float *frc_f,
                   cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(dihedral_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));
  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));

  DihedralForceKernel<<<block_per_grid, thread_per_block, 0, stream>>>(
    dihedral_numbers, uint_crd, scaler, atom_a, atom_b, atom_c, atom_d, ipn, pk, gamc, gams, pn, frc);
  return;
}
void DihedralForce(int dihedral_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f,
                   const int *atom_a, const int *atom_b, const int *atom_c, const int *atom_d, const int *ipn,
                   const float *pk, const float *gamc, const float *gams, const float *pn, float *frc_f,
                   cudaStream_t stream);
