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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/select_impl.cuh"
#include <stdint.h>
#include <limits>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
// Select
template <typename T>
struct SelectFunctor {
  SelectFunctor() {}
  __device__ __forceinline__ T operator()(bool cond, T x, T y) const { return cond ? x : y; }
};
template <typename T>
cudaError_t CalSelect(const bool *cond, const T *input_x, const T *input_y, T *output, const size_t count,
                      cudaStream_t cuda_stream) {
  SelectFunctor<T> functor;
  cuda::elementwise::Ternary(functor, (uint)(count), output, cond, input_x, input_y, cuda_stream);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSelect<double>(const bool *cond, const double *input_x, const double *input_y,
                                                       double *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<float>(const bool *cond, const float *input_x, const float *input_y,
                                                      float *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<half>(const bool *cond, const half *input_x, const half *input_y,
                                                     half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<bool>(const bool *cond, const bool *input_x, const bool *input_y,
                                                     bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<int8_t>(const bool *cond, const int8_t *input_x, const int8_t *input_y,
                                                       int8_t *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<uint8_t>(const bool *cond, const uint8_t *input_x,
                                                        const uint8_t *input_y, uint8_t *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<int16_t>(const bool *cond, const int16_t *input_x,
                                                        const int16_t *input_y, int16_t *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<uint16_t>(const bool *cond, const uint16_t *input_x,
                                                         const uint16_t *input_y, uint16_t *output, const size_t count,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<int32_t>(const bool *cond, const int32_t *input_x,
                                                        const int32_t *input_y, int32_t *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<uint32_t>(const bool *cond, const uint32_t *input_x,
                                                         const uint32_t *input_y, uint32_t *output, const size_t count,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<int64_t>(const bool *cond, const int64_t *input_x,
                                                        const int64_t *input_y, int64_t *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<uint64_t>(const bool *cond, const uint64_t *input_x,
                                                         const uint64_t *input_y, uint64_t *output, const size_t count,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<Complex<float>>(const bool *cond, const Complex<float> *input_x,
                                                               const Complex<float> *input_y, Complex<float> *output,
                                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSelect<Complex<double>>(const bool *cond, const Complex<double> *input_x,
                                                                const Complex<double> *input_y, Complex<double> *output,
                                                                const size_t count, cudaStream_t cuda_stream);
