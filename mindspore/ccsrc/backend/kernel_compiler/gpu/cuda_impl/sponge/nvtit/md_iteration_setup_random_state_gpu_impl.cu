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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_setup_random_state_gpu_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

void MD_Iteration_Setup_Random_State(int float4_numbers, curandStatePhilox4_32_10_t *rand_state, int seed,
                                     cudaStream_t stream) {
  Setup_Rand_Normal_Kernel<<<ceilf(static_cast<float>(float4_numbers) / 32.), 32, 0, stream>>>(float4_numbers,
                                                                                              rand_state, seed);
}

void MD_Iteration_Setup_Random_State(int float4_numbers, curandStatePhilox4_32_10_t *rand_state, int seed,
                                     cudaStream_t stream);
