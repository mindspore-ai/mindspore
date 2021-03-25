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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nb14/dihedral_14_lj_atom_energy_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void Dihedral14LJAtomEnergyKernel(const int dihedral_14_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd,
                                             const VECTOR *boxlength, const int *a_14, const int *b_14,
                                             const float *lj_scale_factor, const float *LJ_type_A,
                                             const float *LJ_type_B, float *ene) {
  int dihedral_14_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (dihedral_14_i < dihedral_14_numbers) {
    int atom_i = a_14[dihedral_14_i];
    int atom_j = b_14[dihedral_14_i];

    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i];
    UINT_VECTOR_LJ_TYPE r2 = uint_crd[atom_j];

    int int_x;
    int int_y;
    int int_z;
    VECTOR dr;
    float dr2;
    float dr_2;
    float dr_4;
    float dr_6;
    float dr_12;
    float ene_lin = 0.;
    int x, y;
    int atom_pair_LJ_type;

    int_x = r2.uint_x - r1.uint_x;
    int_y = r2.uint_y - r1.uint_y;
    int_z = r2.uint_z - r1.uint_z;
    dr.x = boxlength[0].x * int_x;
    dr.y = boxlength[0].y * int_y;
    dr.z = boxlength[0].z * int_z;
    dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

    dr_2 = 1. / dr2;
    dr_4 = dr_2 * dr_2;
    dr_6 = dr_4 * dr_2;
    dr_12 = dr_6 * dr_6;

    y = (r2.LJ_type - r1.LJ_type);
    x = y >> 31;
    y = (y ^ x) - x;
    x = r2.LJ_type + r1.LJ_type;
    r2.LJ_type = (x + y) >> 1;
    x = (x - y) >> 1;
    atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

    ene_lin = 0.08333333 * LJ_type_A[atom_pair_LJ_type] * dr_12 -
              0.1666666 * LJ_type_B[atom_pair_LJ_type] * dr_6;  // LJ的A,B系数已经乘以12和6因此要反乘
    ene_lin *= lj_scale_factor[dihedral_14_i];

    atomicAdd(&ene[atom_i], ene_lin);
  }
}

void Dihedral14LJAtomEnergy(const int dihedral_14_numbers, const int atom_numbers, const int *uint_crd_f,
                            const int *LJtype, const float *charge, const float *boxlength_f, const int *a_14,
                            const int *b_14, const float *lj_scale_factor, const float *LJ_type_A,
                            const float *LJ_type_B, float *ene, cudaStream_t stream) {
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(atom_numbers) / 128);
  UINT_VECTOR_LJ_TYPE *uint_crd_with_LJ = NULL;
  Cuda_Malloc_Safely(reinterpret_cast<void **>(&uint_crd_with_LJ), sizeof(UINT_VECTOR_LJ_TYPE) * atom_numbers);

  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  Copy_Crd_To_New_Crd_Start<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, uint_crd, uint_crd_with_LJ, LJtype, charge);

  VECTOR *boxlength = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(boxlength_f));
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, ene, 0.);
  Dihedral14LJAtomEnergyKernel<<<block_per_grid, thread_per_block, 0, stream>>>(
    dihedral_14_numbers, uint_crd_with_LJ, boxlength, a_14, b_14, lj_scale_factor, LJ_type_A, LJ_type_B, ene);

  cudaStreamSynchronize(stream);

  return;
}

void Dihedral14LJAtomEnergy(const int dihedral_14_numbers, const int atom_numbers, const int *uint_crd_f,
                            const int *LJtype, const float *charge, const float *boxlength_f, const int *a_14,
                            const int *b_14, const float *lj_scale_factor, const float *LJ_type_A,
                            const float *LJ_type_B, float *ene, cudaStream_t stream);
