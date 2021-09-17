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
 *  MDIterationLeapFrog. This is an experimental interface that is subject to change and/or deletion.
 */

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_leap_frog_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void MD_Iteration_Leap_Frog(const int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc,
                                       const float *inverse_mass, const float dt) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    acc[i].x = inverse_mass[i] * frc[i].x;
    acc[i].y = inverse_mass[i] * frc[i].y;
    acc[i].z = inverse_mass[i] * frc[i].z;

    vel[i].x = vel[i].x + dt * acc[i].x;
    vel[i].y = vel[i].y + dt * acc[i].y;
    vel[i].z = vel[i].z + dt * acc[i].z;

    crd[i].x = crd[i].x + dt * vel[i].x;
    crd[i].y = crd[i].y + dt * vel[i].y;
    crd[i].z = crd[i].z + dt * vel[i].z;

    frc[i].x = 0.;
    frc[i].y = 0.;
    frc[i].z = 0.;
  }
}

void MDIterationLeapFrog(const int atom_numbers, float *vel, float *crd, float *frc, float *acc,
                         const float *inverse_mass, const float dt, cudaStream_t stream) {
  VECTOR *d_vel = reinterpret_cast<VECTOR *>(vel);
  VECTOR *d_crd = reinterpret_cast<VECTOR *>(crd);
  VECTOR *d_frc = reinterpret_cast<VECTOR *>(frc);
  VECTOR *d_acc = reinterpret_cast<VECTOR *>(acc);
  MD_Iteration_Leap_Frog<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, d_vel, d_crd, d_frc, d_acc, inverse_mass, dt);
}
