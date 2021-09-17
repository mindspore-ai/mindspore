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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NVTIT_MD_ITERATION_LEAP_FROG_IMPL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NVTIT_MD_ITERATION_LEAP_FROG_IMPL_H

#include <curand_kernel.h>
#include "runtime/device/gpu/cuda_common.h"

void MDIterationLeapFrog(const int atom_numbers, float *vel, float *crd, float *frc, float *acc,
                         const float *inverse_mass, const float dt, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_MD_ITERATION_LEAP_FROG_LIUJIAN_GPU_IMPL_H_
