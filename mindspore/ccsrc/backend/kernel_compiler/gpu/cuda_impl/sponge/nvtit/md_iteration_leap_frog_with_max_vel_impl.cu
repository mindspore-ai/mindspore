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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_leap_frog_with_max_vel_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void MD_Iteration_Leap_Frog_With_Max_Velocity(const int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc,
                                                         VECTOR *acc, const float *inverse_mass, const float dt,
                                                         const float max_velocity) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    VECTOR acc_i = inverse_mass[i] * frc[i];
    VECTOR vel_i = vel[i] + dt * acc_i;
    vel_i = Make_Vector_Not_Exceed_Value(vel_i, max_velocity);
    vel[i] = vel_i;
    crd[i] = crd[i] + dt * vel_i;
    frc[i] = {0.0f, 0.0f, 0.0f};
  }
}

void MDIterationLeapFrogWithMaxVelocity(const int atom_numbers, float *vel, float *crd, float *frc, float *acc,
                                        const float *inverse_mass, const float dt, const float max_velocity,
                                        cudaStream_t stream) {
  VECTOR *d_vel = reinterpret_cast<VECTOR *>(vel);
  VECTOR *d_crd = reinterpret_cast<VECTOR *>(crd);
  VECTOR *d_frc = reinterpret_cast<VECTOR *>(frc);
  VECTOR *d_acc = reinterpret_cast<VECTOR *>(acc);
  MD_Iteration_Leap_Frog_With_Max_Velocity<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, d_vel, d_crd, d_frc, d_acc, inverse_mass, dt, max_velocity);
}
