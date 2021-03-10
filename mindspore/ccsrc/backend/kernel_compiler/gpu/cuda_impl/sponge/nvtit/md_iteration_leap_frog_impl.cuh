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
#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NVTIT_MD_ITERATION_LEAP_FROG_IMPL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NVTIT_MD_ITERATION_LEAP_FROG_IMPL_H

#include <curand_kernel.h>
#include "runtime/device/gpu/cuda_common.h"

void MDIterationLeapFrog(const int float4_numbers, const int atom_numbers, const float half_dt, const float dt,
                         const float exp_gamma, const int is_max_velocity, const float max_velocity,
                         const float *d_mass_inverse, const float *d_sqrt_mass, float *vel_f, float *crd_f,
                         float *frc_f, float *acc_f, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NVTIT_MD_ITERATION_LEAP_FROG_IMPL_H
