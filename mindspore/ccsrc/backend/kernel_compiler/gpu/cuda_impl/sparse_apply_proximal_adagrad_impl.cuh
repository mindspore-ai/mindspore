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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMP_SPARSE_APPLY_PROXIMAL_ADAGRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMP_SPARSE_APPLY_PROXIMAL_ADAGRAD_IMPL_CUH_

#include "runtime/device/gpu/cuda_common.h"
template <typename T>
void CalSparseApplyProximalAdagrad(const size_t size, const size_t indices_size, const T *learning_rate,
                                   const T *l1_regularization, const T *l2_regularization, const T *gradient,
                                   const int *indices, T *variable, T *accumulation, T *variable_out,
                                   T *accumulation_out, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMP_SPARSE_APPLY_PROXIMAL_ADAGRAD_IMPL_CUH_
