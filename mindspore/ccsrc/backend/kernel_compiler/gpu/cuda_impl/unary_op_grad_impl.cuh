/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UNARYOP_GRAD_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UNARYOP_GRAD_IMPL_H_

#include "runtime/device/gpu/cuda_common.h"
template <typename T>
void SqrtGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
void RsqrtGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
void AsinGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
void ACosGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
void AtanGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
void AsinhGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
void AcoshGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
void ReciprocalGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UNARYOP_GRAD_IMPL_H_
