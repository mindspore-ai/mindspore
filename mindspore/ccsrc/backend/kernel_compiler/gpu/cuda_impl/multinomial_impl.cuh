/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MULTINOMIAL_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MULTINOMIAL_IMPL_CUH_
#include <curand_kernel.h>
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
void Multinomial(int seed, int seed2, T *input, int num_sample, curandState *globalState, int *output,
                 size_t distributions, size_t categories, cudaStream_t cuda_stream);
template <typename T>
void CheckNonNeg(const size_t size, const T *input, T *output, cudaStream_t stream);
template <typename T>
void CheckZero(const size_t distributions, const size_t categories, const T *input, T *output, cudaStream_t stream);
template <typename T>
void NormInput(T *input, const size_t distributions, const size_t categories, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MULTINOMIAL_IMPL_CUH_
