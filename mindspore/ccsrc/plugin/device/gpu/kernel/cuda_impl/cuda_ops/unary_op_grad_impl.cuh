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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNARY_OP_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNARY_OP_GRAD_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void SqrtGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void RsqrtGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void AsinGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void ACosGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void AtanGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void TanhGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void SigmoidGrad(const T *input, const T *dout, T *output, const size_t count,
                                 cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void AsinhGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void AcoshGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void ReciprocalGrad(const T *input, const T *dout, T *output, const size_t count,
                                    cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void InvGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNARY_OP_GRAD_IMPL_CUH_
