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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/simple_constrain/refresh_uint_crd_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void Refresh_Uint_Crd(int atom_numbers, const VECTOR *crd, const VECTOR *quarter_crd_to_uint_crd_cof,
                                 UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *test_frc, const float *mass_inverse,
                                 const float half_exp_gamma_plus_half) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    INT_VECTOR tempi;
    VECTOR crd_lin = crd[atom_i];
    VECTOR frc_lin = test_frc[atom_i];
    float mass_lin = mass_inverse[atom_i];

    crd_lin.x = crd_lin.x + half_exp_gamma_plus_half * frc_lin.x * mass_lin;
    crd_lin.y = crd_lin.y + half_exp_gamma_plus_half * frc_lin.y * mass_lin;
    crd_lin.z = crd_lin.z + half_exp_gamma_plus_half * frc_lin.z * mass_lin;

    tempi.int_x = crd_lin.x * quarter_crd_to_uint_crd_cof[0].x;
    tempi.int_y = crd_lin.y * quarter_crd_to_uint_crd_cof[0].y;
    tempi.int_z = crd_lin.z * quarter_crd_to_uint_crd_cof[0].z;

    uint_crd[atom_i].uint_x = tempi.int_x << 2;
    uint_crd[atom_i].uint_y = tempi.int_y << 2;
    uint_crd[atom_i].uint_z = tempi.int_z << 2;
  }
}

void refreshuintcrd(int atom_numbers, float half_exp_gamma_plus_half, const float *crd_f,
                    const float *quarter_crd_to_uint_crd_cof_f, const float *test_frc_f, const float *mass_inverse,
                    unsigned int *uint_crd_f, cudaStream_t stream) {
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(atom_numbers) / 128);
  const VECTOR *crd = reinterpret_cast<const VECTOR *>(crd_f);
  const VECTOR *quarter_crd_to_uint_crd_cof = reinterpret_cast<const VECTOR *>(quarter_crd_to_uint_crd_cof_f);
  const VECTOR *test_frc = reinterpret_cast<const VECTOR *>(test_frc_f);
  UNSIGNED_INT_VECTOR *uint_crd = reinterpret_cast<UNSIGNED_INT_VECTOR *>(uint_crd_f);

  Refresh_Uint_Crd<<<block_per_grid, thread_per_block, 0, stream>>>(atom_numbers, crd,
                                                                    quarter_crd_to_uint_crd_cof,
                                                                    uint_crd, test_frc, mass_inverse,
                                                                    half_exp_gamma_plus_half);
  return;
}

void refreshuintcrd(int atom_numbers, float half_exp_gamma_plus_half, const float *crd_f,
                    const float *quarter_crd_to_uint_crd_cof_f, const float *test_frc_f, const float *mass_inverse,
                    unsigned int *uint_crd_f, cudaStream_t stream);
