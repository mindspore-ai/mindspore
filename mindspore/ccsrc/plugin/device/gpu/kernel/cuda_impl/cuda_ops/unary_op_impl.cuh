/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNARY_OP_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNARY_OP_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
template <typename T>
CUDA_LIB_EXPORT void Exponential(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Expm1(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Logarithm(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Log1p(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Erf(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Erfc(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Negative(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Reciprocal(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Inv(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Invert(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Square(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Sqrt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Rsqrt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Sin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Sinh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Tan(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Cos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Cosh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Asin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void ACos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Atan(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Asinh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Acosh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Atanh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Tanh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Abs(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Floor(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Trunc(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Ceil(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Rint(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Round(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Sign(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Real(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Real(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Imag(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Imag(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Conj(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Conj(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void Sigmoid(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNARY_OP_IMPL_CUH_
