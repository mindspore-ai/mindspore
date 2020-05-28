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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_TAN_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_TAN_H_

#include "device/gpu/cuda_common.h"

template<typename T>
void Tanh(const size_t size, const T* x_addr, T* y_addr, cudaStream_t cuda_stream);

template<typename T>
void TanhGrad(const size_t size, const T* y_addr, const T* dy_addr, T* dx_addr, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_TAN_H_
