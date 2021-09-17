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
/**
 * Note:
 *  ConstrainForceCycleVirial. This is an experimental interface that is subject to change and/or deletion.
 */

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/simple_constrain/constrain_force_virial_impl.cuh"

__global__ void constrain_force_cycle_with_virial_update_kernel(int constrain_pair_numbers,
                                                                const UNSIGNED_INT_VECTOR *uint_crd,
                                                                const VECTOR *scaler, CONSTRAIN_PAIR *constrain_pair,
                                                                const VECTOR *pair_dr, VECTOR *test_frc,
                                                                float *d_atom_virial) {
  int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (pair_i < constrain_pair_numbers) {
    CONSTRAIN_PAIR cp = constrain_pair[pair_i];
    VECTOR dr0 = pair_dr[pair_i];
    VECTOR dr = Get_Periodic_Displacement(uint_crd[cp.atom_i_serial], uint_crd[cp.atom_j_serial], scaler[0]);
    float r_1 = rnorm3df(dr.x, dr.y, dr.z);
    float frc_abs = (1. - cp.constant_r * r_1) * cp.constrain_k;
    VECTOR frc_lin = frc_abs * dr0;
    d_atom_virial[pair_i] -= frc_lin * dr0;

    atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
    atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
    atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

    atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
    atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
    atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
  }
}

__global__ void constrain_force_cycle_update_kernel(const int constrain_pair_numbers,
                                                    const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *scaler,
                                                    const CONSTRAIN_PAIR *constrain_pair, const VECTOR *pair_dr,
                                                    VECTOR *test_frc) {
  int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (pair_i < constrain_pair_numbers) {
    CONSTRAIN_PAIR cp = constrain_pair[pair_i];
    float r_1;
    VECTOR dr;
    float frc_abs;
    VECTOR frc_lin;

    dr.x = (static_cast<int>(uint_crd[cp.atom_i_serial].uint_x - uint_crd[cp.atom_j_serial].uint_x)) * scaler[0].x;
    dr.y = (static_cast<int>(uint_crd[cp.atom_i_serial].uint_y - uint_crd[cp.atom_j_serial].uint_y)) * scaler[0].y;
    dr.z = (static_cast<int>(uint_crd[cp.atom_i_serial].uint_z - uint_crd[cp.atom_j_serial].uint_z)) * scaler[0].z;
    r_1 = rnorm3df(dr.x, dr.y, dr.z);
    frc_abs = (1. - cp.constant_r * r_1) * cp.constrain_k;

    frc_lin.x = frc_abs * pair_dr[pair_i].x;
    frc_lin.y = frc_abs * pair_dr[pair_i].y;
    frc_lin.z = frc_abs * pair_dr[pair_i].z;

    atomicAdd(&test_frc[cp.atom_j_serial].x, frc_lin.x);
    atomicAdd(&test_frc[cp.atom_j_serial].y, frc_lin.y);
    atomicAdd(&test_frc[cp.atom_j_serial].z, frc_lin.z);

    atomicAdd(&test_frc[cp.atom_i_serial].x, -frc_lin.x);
    atomicAdd(&test_frc[cp.atom_i_serial].y, -frc_lin.y);
    atomicAdd(&test_frc[cp.atom_i_serial].z, -frc_lin.z);
  }
}

__global__ void refresh_uint_crd_update_kernel(int atom_numbers, const VECTOR *crd,
                                               const VECTOR *quarter_crd_to_uint_crd_cof, UNSIGNED_INT_VECTOR *uint_crd,
                                               VECTOR *test_frc, const float *mass_inverse,
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

void constrain_force_cycle_update(int atom_numbers, int constrain_pair_numbers, const unsigned int *uint_crd_f,
                                  const float *scaler_f, float *constrain_pair_f, const float *pair_dr_f,
                                  const int *atom_i_serials, const int *atom_j_serials, const float *constant_rs,
                                  const float *constrain_ks, float *test_frc_f, cudaStream_t stream) {
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(atom_numbers) / 128);
  const UNSIGNED_INT_VECTOR *uint_crd = reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f);
  const VECTOR *scaler = reinterpret_cast<const VECTOR *>(scaler_f);
  const VECTOR *pair_dr = reinterpret_cast<const VECTOR *>(pair_dr_f);

  VECTOR *test_frc = reinterpret_cast<VECTOR *>(test_frc_f);

  CONSTRAIN_PAIR *constrain_pair = reinterpret_cast<CONSTRAIN_PAIR *>(constrain_pair_f);

  construct_constrain_pair<<<ceilf(static_cast<float>(constrain_pair_numbers) / 128), 128, 0, stream>>>(
    constrain_pair_numbers, atom_i_serials, atom_j_serials, constant_rs, constrain_ks, constrain_pair);

  constrain_force_cycle_update_kernel<<<block_per_grid, thread_per_block, 0, stream>>>(
    constrain_pair_numbers, uint_crd, scaler, constrain_pair, pair_dr, test_frc);

  return;
}

void constrain_force_cycle_with_virial_update(int atom_numbers, int constrain_pair_numbers,
                                              const unsigned int *uint_crd_f, const float *scaler_f,
                                              float *constrain_pair_f, const float *pair_dr_f,
                                              const int *atom_i_serials, const int *atom_j_serials,
                                              const float *constant_rs, const float *constrain_ks, float *test_frc_f,
                                              float *d_atom_virial, cudaStream_t stream) {
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(atom_numbers) / 128);
  const UNSIGNED_INT_VECTOR *uint_crd = reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f);
  const VECTOR *scaler = reinterpret_cast<const VECTOR *>(scaler_f);
  const VECTOR *pair_dr = reinterpret_cast<const VECTOR *>(pair_dr_f);

  VECTOR *test_frc = reinterpret_cast<VECTOR *>(test_frc_f);

  CONSTRAIN_PAIR *constrain_pair = reinterpret_cast<CONSTRAIN_PAIR *>(constrain_pair_f);

  construct_constrain_pair<<<ceilf(static_cast<float>(constrain_pair_numbers) / 128), 128, 0, stream>>>(
    constrain_pair_numbers, atom_i_serials, atom_j_serials, constant_rs, constrain_ks, constrain_pair);

  constrain_force_cycle_with_virial_update_kernel<<<block_per_grid, thread_per_block, 0, stream>>>(
    constrain_pair_numbers, uint_crd, scaler, constrain_pair, pair_dr, test_frc, d_atom_virial);

  return;
}

void refresh_uint_crd_update(int atom_numbers, float half_exp_gamma_plus_half, const float *crd_f,
                             const float *quarter_crd_to_uint_crd_cof_f, float *test_frc_f, const float *mass_inverse,
                             unsigned int *uint_crd_f, cudaStream_t stream) {
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(atom_numbers) / 128);
  const VECTOR *crd = reinterpret_cast<const VECTOR *>(crd_f);
  const VECTOR *quarter_crd_to_uint_crd_cof = reinterpret_cast<const VECTOR *>(quarter_crd_to_uint_crd_cof_f);
  VECTOR *test_frc = reinterpret_cast<VECTOR *>(test_frc_f);
  UNSIGNED_INT_VECTOR *uint_crd = reinterpret_cast<UNSIGNED_INT_VECTOR *>(uint_crd_f);

  refresh_uint_crd_update_kernel<<<block_per_grid, thread_per_block, 0, stream>>>(
    atom_numbers, crd, quarter_crd_to_uint_crd_cof, uint_crd, test_frc, mass_inverse, half_exp_gamma_plus_half);
  return;
}

void set_zero_force_with_virial(int atom_numbers, int constrain_pair_numbers, float *test_frc_f, float *d_atom_virial,
                                cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3 * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, test_frc_f, 0.);
  Reset_List<<<ceilf(static_cast<float>(constrain_pair_numbers) / 128), 128, 0, stream>>>(constrain_pair_numbers,
                                                                                          d_atom_virial, 0.);
  return;
}

void set_zero(int numbers, float *x, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(numbers) / 128), 128, 0, stream>>>(numbers, x, 0.);
  return;
}
