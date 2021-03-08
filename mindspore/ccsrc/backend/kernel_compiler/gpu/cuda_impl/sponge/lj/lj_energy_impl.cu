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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/lj/lj_energy_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void LJ_Energy_CUDA(const int atom_numbers, const NEIGHBOR_LIST *nl, const UINT_VECTOR_LJ_TYPE *uint_crd,
                               const VECTOR *boxlength, const float *LJ_type_A, const float *LJ_type_B,
                               const float cutoff_square, float *lj_ene) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    NEIGHBOR_LIST nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2;
    float dr_2;
    float dr_4;
    float dr_6;
    float ene_lin = 0.;

    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength[0].x * int_x;
      dr.y = boxlength[0].y * int_y;
      dr.z = boxlength[0].z * int_z;

      dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
      if (dr2 < cutoff_square) {
        dr_2 = 1. / dr2;
        dr_4 = dr_2 * dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        dr_2 = (0.083333333 * LJ_type_A[atom_pair_LJ_type] * dr_6 - 0.166666666 * LJ_type_B[atom_pair_LJ_type]) * dr_6;
        ene_lin = ene_lin + dr_2;
      }
    }
    atomicAdd(&lj_ene[atom_i], ene_lin);
  }
}

void LJEnergy(const int atom_numbers, const float cutoff_square, const int *uint_crd_f, const int *LJtype,
              const float *charge, const float *scaler_f, float *uint_crd_with_LJ, int *nl_atom_numbers,
              int *nl_atom_serial, int *nl, const float *d_LJ_A, const float *d_LJ_B, float *d_LJ_energy_atom,
              cudaStream_t stream) {
  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));
  int max_neighbor_numbers = 800;
  NEIGHBOR_LIST *nl_a = reinterpret_cast<NEIGHBOR_LIST *>(nl);
  construct_neighbor_list_kernel<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, max_neighbor_numbers, nl_atom_numbers, nl_atom_serial, nl_a);

  UINT_VECTOR_LJ_TYPE *uint_crd_with_LJ_a = reinterpret_cast<UINT_VECTOR_LJ_TYPE *>(uint_crd_with_LJ);

  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  Copy_Crd_To_New_Crd_Start<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, uint_crd, uint_crd_with_LJ_a, LJtype, charge);

  Reset_List<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, d_LJ_energy_atom, 0.);

  LJ_Energy_CUDA<<<ceilf(static_cast<float>(atom_numbers) / 8), thread_LJ, 0, stream>>>(
    atom_numbers, nl_a, uint_crd_with_LJ_a, scaler, d_LJ_A, d_LJ_B, cutoff_square, d_LJ_energy_atom);

  return;
}
void LJEnergy(const int atom_numbers, const float cutoff_square, const int *uint_crd_f, const int *LJtype,
              const float *charge, const float *scaler_f, float *uint_crd_with_LJ, int *nl_atom_numbers,
              int *nl_atom_serial, int *nl, const float *d_LJ_A, const float *d_LJ_B, float *d_LJ_energy_atom,
              cudaStream_t stream);
