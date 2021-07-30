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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_leap_frog_liujian_with_max_vel_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void MD_Iteration_Leap_Frog_With_LiuJian_With_Max_Velocity(
  const int atom_numbers, const float half_dt, const float dt, const float exp_gamma, const float *inverse_mass,
  const float *sqrt_mass_inverse, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, VECTOR *random_frc,
  VECTOR *output, const float max_vel) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  float abs_vel;
  if (i < atom_numbers) {
    acc[i].x = inverse_mass[i] * frc[i].x;
    acc[i].y = inverse_mass[i] * frc[i].y;
    acc[i].z = inverse_mass[i] * frc[i].z;

    vel[i].x = vel[i].x + dt * acc[i].x;
    vel[i].y = vel[i].y + dt * acc[i].y;
    vel[i].z = vel[i].z + dt * acc[i].z;

    abs_vel = norm3df(vel[i].x, vel[i].y, vel[i].z);
    if (abs_vel < max_vel) {
    } else {
      abs_vel = max_vel / abs_vel;
      vel[i].x = abs_vel * vel[i].x;
      vel[i].y = abs_vel * vel[i].y;
      vel[i].z = abs_vel * vel[i].z;
    }

    output[i].x = crd[i].x + half_dt * vel[i].x;
    output[i].y = crd[i].y + half_dt * vel[i].y;
    output[i].z = crd[i].z + half_dt * vel[i].z;

    vel[i].x = exp_gamma * vel[i].x + sqrt_mass_inverse[i] * random_frc[i].x;
    vel[i].y = exp_gamma * vel[i].y + sqrt_mass_inverse[i] * random_frc[i].y;
    vel[i].z = exp_gamma * vel[i].z + sqrt_mass_inverse[i] * random_frc[i].z;

    output[i].x = output[i].x + half_dt * vel[i].x;
    output[i].y = output[i].y + half_dt * vel[i].y;
    output[i].z = output[i].z + half_dt * vel[i].z;

    frc[i].x = 0.;
    frc[i].y = 0.;
    frc[i].z = 0.;
  }
}
void MD_Iteration_Leap_Frog_With_LiuJian_With_Max_Vel(const int atom_numbers, const float half_dt, const float dt,
                                                      const float exp_gamma, int float4_numbers, float *inverse_mass,
                                                      float *sqrt_mass_inverse, float *vel, float *crd, float *frc,
                                                      float *acc, curandStatePhilox4_32_10_t *rand_state,
                                                      float *rand_frc, float *output, const float max_vel,
                                                      cudaStream_t stream) {
  Rand_Normal<<<ceilf(static_cast<float>(float4_numbers) / 32.), 32, 0, stream>>>(float4_numbers, rand_state,
                                                                                  reinterpret_cast<float4 *>(rand_frc));
  VECTOR *d_vel = reinterpret_cast<VECTOR *>(vel);
  VECTOR *d_crd = reinterpret_cast<VECTOR *>(crd);
  VECTOR *d_frc = reinterpret_cast<VECTOR *>(frc);
  VECTOR *d_acc = reinterpret_cast<VECTOR *>(acc);
  VECTOR *d_rand_frc = reinterpret_cast<VECTOR *>(rand_frc);
  VECTOR *d_out = reinterpret_cast<VECTOR *>(output);
  MD_Iteration_Leap_Frog_With_LiuJian_With_Max_Velocity<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0,
                                                          stream>>>(atom_numbers, half_dt, dt, exp_gamma, inverse_mass,
                                                                    sqrt_mass_inverse, d_vel, d_crd, d_frc, d_acc,
                                                                    d_rand_frc, d_out, max_vel);
}
