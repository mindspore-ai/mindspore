/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_PADIMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_PADIMPL_H_
#include <cuda_runtime.h>
#include "device/gpu/cuda_common.h"

template <typename T>
void CalPad(const size_t size, const T* input, const int num, const int channels, const int old_height,
            const int old_width, const int padded_height, const int padded_width, const int pad_top, const int pad_left,
            float pad_value, T* output, cudaStream_t cuda_stream);
template <typename T>
void CalPadGrad(const size_t size, const T* dy, const int num, const int channels, const int old_height,
                const int old_width, const int padded_height, const int padded_width, const int pad_top,
                const int pad_left, T* dx, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_PADIMPL_H_
