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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_SQUARE_SUM_ALL_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_SQUARE_SUM_ALL_IMPL_H_

#include "runtime/device/gpu/cuda_common.h"
template <typename T>
void SquareSumAll(const size_t input_size_, const T* input_addr_0, const T* input_addr_1,
                  T* output_addr_0, T* output_addr_1, float* ws_addr_0, float* ws_addr_1, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_SQUARE_SUM_ALL_IMPL_H_
