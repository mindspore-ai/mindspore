/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OP_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OP_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
CUDA_LIB_EXPORT void ExpOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void LogOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void NegOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void ReciprocalOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void InvOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void SquareOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void SqrtOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void CalOnesLike(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void LogicalNot(const T *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void CalSelect(const bool *cond, const T *input_x, const T *input_y, T *output, const size_t count,
                               cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void CalReLU(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void TanhOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void SigmoidOpt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void TanhGradOpt(const T *input, const T *dout, T *output, const size_t count,
                                 cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void SigmoidGradOpt(const T *input, const T *dout, T *output, const size_t count,
                                    cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void SiLUGradOpt(const T *input, const T *dout, T *output, const size_t count,
                                    cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OP_IMPL_CUH_
