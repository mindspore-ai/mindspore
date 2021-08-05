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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/simple_constrain/refresh_crd_vel_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void Refresh_Crd_Vel(int atom_numbers, float dt_inverse, float dt, float exp_gamma,
                                float half_exp_gamma_plus_half, VECTOR *test_frc, float *mass_inverse, VECTOR *crd,
                                VECTOR *vel) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    VECTOR crd_lin = crd[atom_i];
    VECTOR frc_lin = test_frc[atom_i];
    VECTOR vel_lin = vel[atom_i];
    float mass_lin = mass_inverse[atom_i];

    frc_lin.x = frc_lin.x * mass_lin;
    frc_lin.y = frc_lin.y * mass_lin;
    frc_lin.z = frc_lin.z * mass_lin;  // mass实际为mass的倒数，frc_lin已经乘以dt^2

    crd_lin.x = crd_lin.x + half_exp_gamma_plus_half * frc_lin.x;
    crd_lin.y = crd_lin.y + half_exp_gamma_plus_half * frc_lin.y;
    crd_lin.z = crd_lin.z + half_exp_gamma_plus_half * frc_lin.z;

    vel_lin.x = (vel_lin.x + exp_gamma * frc_lin.x * dt_inverse);
    vel_lin.y = (vel_lin.y + exp_gamma * frc_lin.y * dt_inverse);
    vel_lin.z = (vel_lin.z + exp_gamma * frc_lin.z * dt_inverse);

    crd[atom_i] = crd_lin;
    vel[atom_i] = vel_lin;
  }
}

void refreshcrdvel(int atom_numbers, float dt_inverse, float dt, float exp_gamma, float half_exp_gamma_plus_half,
                   float *test_frc_f, float *mass_inverse, float *crd_f, float *vel_f, cudaStream_t stream) {
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(atom_numbers) / 128);
  VECTOR *crd = reinterpret_cast<VECTOR *>(crd_f);
  VECTOR *vel = reinterpret_cast<VECTOR *>(vel_f);
  VECTOR *test_frc = reinterpret_cast<VECTOR *>(test_frc_f);

  Refresh_Crd_Vel<<<block_per_grid, thread_per_block, 0, stream>>>(
    atom_numbers, dt_inverse, dt, exp_gamma, half_exp_gamma_plus_half, test_frc, mass_inverse, crd, vel);
  return;
}

void refreshcrdvel(int atom_numbers, float dt_inverse, float dt, float exp_gamma, float half_exp_gamma_plus_half,
                   float *test_frc_f, float *mass_inverse, float *crd_f, float *vel_f, cudaStream_t stream);
