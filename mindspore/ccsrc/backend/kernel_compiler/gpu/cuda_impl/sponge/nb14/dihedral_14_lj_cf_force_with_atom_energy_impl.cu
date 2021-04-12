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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nb14/dihedral_14_lj_cf_force_with_atom_energy_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void Dihedral14LJCFForceWithAtomEnergyKernel(const int dihedral_14_numbers,
                                                        const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR *boxlength,
                                                        const int *a_14, const int *b_14, const float *lj_scale_factor,
                                                        const float *cf_scale_factor, const float *LJ_type_A,
                                                        const float *LJ_type_B, VECTOR *frc, float *atom_energy) {
  int dihedral_14_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (dihedral_14_i < dihedral_14_numbers) {
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1, r2;
    VECTOR dr;
    float dr_abs;
    float dr2;
    float dr_1;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_14;
    float frc_abs = 0.;
    VECTOR temp_frc;

    float ene_lin;
    float ene_lin2;

    int x, y;
    int atom_pair_LJ_type;

    int atom_i = a_14[dihedral_14_i];
    int atom_j = b_14[dihedral_14_i];

    r1 = uint_crd[atom_i];
    r2 = uint_crd[atom_j];
    int_x = r2.uint_x - r1.uint_x;
    int_y = r2.uint_y - r1.uint_y;
    int_z = r2.uint_z - r1.uint_z;
    dr.x = boxlength[0].x * int_x;
    dr.y = boxlength[0].y * int_y;
    dr.z = boxlength[0].z * int_z;
    dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

    dr_2 = 1.0 / dr2;
    dr_4 = dr_2 * dr_2;
    dr_8 = dr_4 * dr_4;
    dr_14 = dr_8 * dr_4 * dr_2;
    dr_abs = norm3df(dr.x, dr.y, dr.z);
    dr_1 = 1. / dr_abs;

    float charge_i = r1.charge;
    float charge_j = r2.charge;
    float frc_cf_abs;
    frc_cf_abs = cf_scale_factor[dihedral_14_i] * dr_2 * dr_1;
    frc_cf_abs = -charge_i * charge_j * frc_cf_abs;

    y = (r2.LJ_type - r1.LJ_type);
    x = y >> 31;
    y = (y ^ x) - x;
    x = r2.LJ_type + r1.LJ_type;
    r2.LJ_type = (x + y) >> 1;
    x = (x - y) >> 1;
    atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

    frc_abs = -LJ_type_A[atom_pair_LJ_type] * dr_14 + LJ_type_B[atom_pair_LJ_type] * dr_8;
    frc_abs *= lj_scale_factor[dihedral_14_i];

    frc_abs += frc_cf_abs;
    temp_frc.x = frc_abs * dr.x;
    temp_frc.y = frc_abs * dr.y;
    temp_frc.z = frc_abs * dr.z;

    atomicAdd(&frc[atom_j].x, -temp_frc.x);
    atomicAdd(&frc[atom_j].y, -temp_frc.y);
    atomicAdd(&frc[atom_j].z, -temp_frc.z);
    atomicAdd(&frc[atom_i].x, temp_frc.x);
    atomicAdd(&frc[atom_i].y, temp_frc.y);
    atomicAdd(&frc[atom_i].z, temp_frc.z);

    ene_lin = r1.charge * r2.charge * dr_1;
    ene_lin *= cf_scale_factor[dihedral_14_i];
    ene_lin2 = 0.08333333 * LJ_type_A[atom_pair_LJ_type] * dr_4 * dr_8 -
               0.1666666 * LJ_type_B[atom_pair_LJ_type] * dr_4 * dr_2;  // LJ的A,B系数已经乘以12和6因此要反乘
    ene_lin2 *= lj_scale_factor[dihedral_14_i];

    atomicAdd(&atom_energy[atom_i], ene_lin + ene_lin2);
  }
}

void Dihedral14LJCFForceWithAtomEnergy(const int dihedral_14_numbers, const int atom_numbers, const int *uint_crd_f,
                                       const int *LJtype, const float *charge, float *uint_crd_with_LJ_f,
                                       const float *boxlength_f, const int *a_14, const int *b_14,
                                       const float *lj_scale_factor, const float *cf_scale_factor,
                                       const float *LJ_type_A, const float *LJ_type_B, float *frc_f, float *atom_energy,
                                       cudaStream_t stream) {
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(dihedral_14_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  UINT_VECTOR_LJ_TYPE *uint_crd_with_LJ = reinterpret_cast<UINT_VECTOR_LJ_TYPE *>(uint_crd_with_LJ_f);

  Copy_Crd_To_New_Crd_Start<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, uint_crd, uint_crd_with_LJ, LJtype, charge);

  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  Reset_List<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, atom_energy, 0.);
  VECTOR *boxlength = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(boxlength_f));
  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));

  Dihedral14LJCFForceWithAtomEnergyKernel<<<block_per_grid, thread_per_block, 0, stream>>>(
    dihedral_14_numbers, uint_crd_with_LJ, boxlength, a_14, b_14, lj_scale_factor, cf_scale_factor, LJ_type_A,
    LJ_type_B, frc, atom_energy);

  return;
}

void Dihedral14LJCFForceWithAtomEnergy(const int dihedral_14_numbers, const int atom_numbers, const int *uint_crd_f,
                                       const int *LJtype, const float *charge, float *uint_crd_with_LJ_f,
                                       const float *boxlength_f, const int *a_14, const int *b_14,
                                       const float *lj_scale_factor, const float *cf_scale_factor,
                                       const float *LJ_type_A, const float *LJ_type_B, float *frc_f, float *atom_energy,
                                       cudaStream_t stream);
