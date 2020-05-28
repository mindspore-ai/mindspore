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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SLICEIMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SLICEIMPL_H_

#include <cuda_runtime.h>
#include <vector>
#include "device/gpu/cuda_common.h"


template <typename T>
void Slice4DKernel(const int s1, const int s2, const int s3, const int s4,
                   const int l1, const int l2, const int l3, const int l4,
                   const int d1, const int d2, const int d3, const int d4,
                   const T *input, T *output, cudaStream_t stream);
template <typename T>
void CalSliceGrad(const size_t input_size, const T* input, const std::vector<int> in_shape,
                  const std::vector<int> begin, const std::vector<int> size, T* output, cudaStream_t cuda_stream);
template <typename T>
void CalStridedSlice(const size_t input_size, const T* input, const std::vector<int> in_shape,
                     const std::vector<int> begin, const std::vector<int> end, const std::vector<int> strides,
                     T* output, cudaStream_t cuda_stream);
template <typename T>
void CalStridedSliceGrad(const size_t input_size, const T* dy, const std::vector<int> in_shape,
                         const std::vector<int> begin, const std::vector<int> end, const std::vector<int> strides,
                         T* dx, cudaStream_t cuda_stream);
template <typename T>
void FillDeviceArray(const size_t input_size, T* addr, const float value, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SLICEIMPL_H_
