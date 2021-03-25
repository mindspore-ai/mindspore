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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_leap_frog_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void MD_Iteration_Leap_Frog_With_LiuJian(const int atom_numbers, const float half_dt, const float dt,
                                                    const float exp_gamma, const float *inverse_mass,
                                                    const float *sqrt_mass_inverse, VECTOR *vel, VECTOR *crd,
                                                    VECTOR *frc, VECTOR *acc, VECTOR *random_frc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    acc[i].x = inverse_mass[i] * frc[i].x;
    acc[i].y = inverse_mass[i] * frc[i].y;
    acc[i].z = inverse_mass[i] * frc[i].z;

    vel[i].x = vel[i].x + dt * acc[i].x;
    vel[i].y = vel[i].y + dt * acc[i].y;
    vel[i].z = vel[i].z + dt * acc[i].z;

    crd[i].x = crd[i].x + half_dt * vel[i].x;
    crd[i].y = crd[i].y + half_dt * vel[i].y;
    crd[i].z = crd[i].z + half_dt * vel[i].z;

    vel[i].x = exp_gamma * vel[i].x + sqrt_mass_inverse[i] * random_frc[i].x;
    vel[i].y = exp_gamma * vel[i].y + sqrt_mass_inverse[i] * random_frc[i].y;
    vel[i].z = exp_gamma * vel[i].z + sqrt_mass_inverse[i] * random_frc[i].z;

    crd[i].x = crd[i].x + half_dt * vel[i].x;
    crd[i].y = crd[i].y + half_dt * vel[i].y;
    crd[i].z = crd[i].z + half_dt * vel[i].z;

    frc[i].x = 0.;
    frc[i].y = 0.;
    frc[i].z = 0.;
  }
}

__global__ void MD_Iteration_Leap_Frog_With_LiuJian_With_Max_Velocity(const int atom_numbers, const float half_dt,
                                                                      const float dt, const float exp_gamma,
                                                                      const float *inverse_mass,
                                                                      const float *sqrt_mass_inverse, VECTOR *vel,
                                                                      VECTOR *crd, VECTOR *frc, VECTOR *acc,
                                                                      VECTOR *random_frc, const float max_vel) {
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

    crd[i].x = crd[i].x + half_dt * vel[i].x;
    crd[i].y = crd[i].y + half_dt * vel[i].y;
    crd[i].z = crd[i].z + half_dt * vel[i].z;

    vel[i].x = exp_gamma * vel[i].x + sqrt_mass_inverse[i] * random_frc[i].x;
    vel[i].y = exp_gamma * vel[i].y + sqrt_mass_inverse[i] * random_frc[i].y;
    vel[i].z = exp_gamma * vel[i].z + sqrt_mass_inverse[i] * random_frc[i].z;

    crd[i].x = crd[i].x + half_dt * vel[i].x;
    crd[i].y = crd[i].y + half_dt * vel[i].y;
    crd[i].z = crd[i].z + half_dt * vel[i].z;

    frc[i].x = 0.;
    frc[i].y = 0.;
    frc[i].z = 0.;
  }
}

void MDIterationLeapFrog(const int float4_numbers, const int atom_numbers, const float half_dt, const float dt,
                         const float exp_gamma, const int is_max_velocity, const float max_velocity,
                         const float *d_mass_inverse, const float *d_sqrt_mass, float *vel_f, float *crd_f,
                         float *frc_f, float *acc_f, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128>>>(3 * atom_numbers, acc_f, 0.);

  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));
  VECTOR *vel = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(vel_f));
  VECTOR *acc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(acc_f));
  VECTOR *crd = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(crd_f));

  curandStatePhilox4_32_10_t *rand_state;
  VECTOR *random_force;

  Cuda_Malloc_Safely(reinterpret_cast<void **>(&random_force), sizeof(float4) * float4_numbers);
  Cuda_Malloc_Safely(reinterpret_cast<void **>(&rand_state), sizeof(curandStatePhilox4_32_10_t) * float4_numbers);
  Setup_Rand_Normal_Kernel<<<ceilf(static_cast<float>(float4_numbers) / 32.), 32>>>(float4_numbers, rand_state, 1);
  Rand_Normal<<<ceilf(static_cast<float>(float4_numbers) / 32.), 32, 0, stream>>>(
    float4_numbers, rand_state, reinterpret_cast<float4 *>(random_force));

  if (!is_max_velocity) {
    MD_Iteration_Leap_Frog_With_LiuJian<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
      atom_numbers, half_dt, dt, exp_gamma, d_mass_inverse, d_sqrt_mass, vel, crd, frc, acc, random_force);
  } else {
    MD_Iteration_Leap_Frog_With_LiuJian_With_Max_Velocity<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0,
                                                            stream>>>(atom_numbers, half_dt, dt, exp_gamma,
                                                                      d_mass_inverse, d_sqrt_mass, vel, crd, frc, acc,
                                                                      random_force, max_velocity);

    cudaStreamSynchronize(stream);
    cudaFree(random_force);
    cudaFree(rand_state);

    return;
  }
}

void MDIterationLeapFrog(const int float4_numbers, const int atom_numbers, const float half_dt, const float dt,
                         const float exp_gamma, const int is_max_velocity, const float max_velocity,
                         const float *d_mass_inverse, const float *d_sqrt_mass, float *vel_f, float *crd_f,
                         float *frc_f, float *acc_f, cudaStream_t stream);
