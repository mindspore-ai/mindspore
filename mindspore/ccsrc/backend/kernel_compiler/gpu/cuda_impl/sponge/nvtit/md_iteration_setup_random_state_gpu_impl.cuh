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
#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_MD_ITERATION_SETUP_RANDOM_STATE_GPU_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_MD_ITERATION_SETUP_RANDOM_STATE_GPU_IMPL_H_

#include <curand_kernel.h>
#include "runtime/device/gpu/cuda_common.h"
void MD_Iteration_Setup_Random_State(int float4_numbers, curandStatePhilox4_32_10_t *rand_state, int seed,
                                     cudaStream_t stream);
#endif
