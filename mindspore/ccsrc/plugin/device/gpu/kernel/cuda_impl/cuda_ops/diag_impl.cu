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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/diag_impl.cuh"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

// Assume input has dimensions :math:`[D_1,... D_k]`, the output is a tensor of
// rank 2k with dimensions :math:`[D_1,..., D_k, D_1,..., D_k]` where
// :math:`output[i_1,..., i_k, i_1,..., i_k] = input_x[i_1,..., i_k]` and 0 everywhere else.
//
// :math:`input_size = D_1*...*D_k`
// :math:`output_size = input_size * input_size`
// :math:`input_index = i_1*(D_2*...*D_k) + i_2*(D_3*...*D_k) +...+ i_k`
// :math:`output_index = i_1*(D_2*...*D_k*D_1*...*D_k) + i_2*(D_3*...*D_k*D_1*...*D_k) +...+ i_k*(D_1*...*D_k) +
//                                              i_1*(D_2*...*D_k) + i_2*(D_3*...*D_k) +...+ i_k
//                                          = (i_1*(D_2*...*D_k) + i_2*(D_3*...*D_k) +...+ i_k)*(D_1*...*D_k) +
//                                              i_1*(D_2*...*D_k) + i_2*(D_3*...*D_k) +...+ i_k
//                                          = input_index*(D_1*...*D_k) + input_index
//                                          = input_index*(input_size + 1) `
template <typename DataType>
__global__ void DiagKernel(const DataType *input_ptr, DataType *output_ptr, size_t input_size, size_t output_size) {
  for (size_t output_index = blockIdx.x * blockDim.x + threadIdx.x; output_index < output_size;
       output_index += blockDim.x * gridDim.x) {
    if (output_index % (input_size + 1) == 0) {
      output_ptr[output_index] = input_ptr[output_index / (input_size + 1)];
    } else {
      output_ptr[output_index] = 0;
    }
  }
}

template <typename DataType>
void CalDiag(const DataType *input_ptr, DataType *output_ptr, size_t input_size, size_t output_size,
             cudaStream_t cuda_stream) {
  DiagKernel<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(input_ptr, output_ptr, input_size, output_size);
}

template CUDA_LIB_EXPORT void CalDiag<uint8_t>(const uint8_t *input_ptr, uint8_t *output_ptr, size_t input_size,
                                               size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<uint16_t>(const uint16_t *input_ptr, uint16_t *output_ptr, size_t input_size,
                                                size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<uint32_t>(const uint32_t *input_ptr, uint32_t *output_ptr, size_t input_size,
                                                size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<uint64_t>(const uint64_t *input_ptr, uint64_t *output_ptr, size_t input_size,
                                                size_t output_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalDiag<half>(const half *input_ptr, half *output_ptr, size_t input_size,
                                            size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<float>(const float *input_ptr, float *output_ptr, size_t input_size,
                                             size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<double>(const double *input_ptr, double *output_ptr, size_t input_size,
                                              size_t output_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalDiag<int8_t>(const int8_t *input_ptr, int8_t *output_ptr, size_t input_size,
                                              size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<int16_t>(const int16_t *input_ptr, int16_t *output_ptr, size_t input_size,
                                               size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<int32_t>(const int32_t *input_ptr, int32_t *output_ptr, size_t input_size,
                                               size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<int64_t>(const int64_t *input_ptr, int64_t *output_ptr, size_t input_size,
                                               size_t output_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalDiag<Complex<float>>(const Complex<float> *input_ptr, Complex<float> *output_ptr,
                                               size_t input_size, size_t output_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDiag<Complex<double>>(const Complex<double> *input_ptr, Complex<double> *output_ptr,
                                               size_t input_size, size_t output_size, cudaStream_t cuda_stream);
