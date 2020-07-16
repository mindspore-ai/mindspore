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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_RESIZE_NEAREST_NEIGHBOR_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_RESIZE_NEAREST_NEIGHBOR_IMPL_CUH_
#include <cuda_runtime.h>
#include "runtime/device/gpu/cuda_common.h"
#define RESIZENEARESTNEIGHBOR_DIMENSION 4

template <typename T>
void CalResizeNearestNeighbor(const int size, const T *input, const int s1, const int s2, const int s3, const int s4,
                              T *output, const int d1, const int d2, const int d3, const int d4, bool align_corners,
                              float h_scale, float w_scale, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_RESIZE_NEAREST_NEIGHBOR_IMPL_CUH_
