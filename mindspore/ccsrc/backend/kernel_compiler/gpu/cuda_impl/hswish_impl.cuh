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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_HSWISH_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_HSWISH_IMPL_CUH_

#include <cuda_runtime.h>
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
void CalHSwish(const size_t &size, const T *input, T *output, cudaStream_t cuda_stream);

template <typename T>
void CalHSwishGrad(const size_t &size, const T *dout, const T *x, T *output, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_HSWISH_IMPL_CUH_
