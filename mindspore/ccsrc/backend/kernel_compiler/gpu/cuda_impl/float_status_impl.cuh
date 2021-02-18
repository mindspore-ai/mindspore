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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_FLOATSTATUS_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_FLOATSTATUS_H_
#include "runtime/device/gpu/cuda_common.h"
template <typename T>
void CalFloatStatus(const size_t size, const T *input, float *output, cudaStream_t stream);
template <typename T>
void CalIsNan(const size_t size, const T *input, bool *output, cudaStream_t stream);
template <typename T>
void CalIsInf(const size_t size, const T *input, bool *output, cudaStream_t stream);
template <typename T>
void CalIsFinite(const size_t size, const T *input, bool *output, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_FLOATSTATUS_H_
