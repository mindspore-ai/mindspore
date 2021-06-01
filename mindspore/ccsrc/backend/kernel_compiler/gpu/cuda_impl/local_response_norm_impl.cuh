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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_LOCAL_RESPONSE_NORM_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_LOCAL_RESPONSE_NORM_H_
#include "runtime/device/gpu/cuda_common.h"
template <typename T>
void CalLocalResponseNormNHWC(const T *input, const int depth_radius, const float bias, const float alpha,
  const float beta, const size_t channels, const size_t num_elements, float *scale, T *output,
  cudaStream_t cuda_stream);

template <typename T>
void CalLocalResponseNormGradNHWC(const T *dy, const T *x, const T *y, const int depth_radius, const float bias,
  const float alpha, const float beta, const size_t channels, const size_t num_elements, float *scale, T *dx,
  cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_LOCAL_RESPONSE_NORM_H_
