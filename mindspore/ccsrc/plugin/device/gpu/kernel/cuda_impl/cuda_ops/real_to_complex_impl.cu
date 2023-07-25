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

#include <cuda_runtime.h>
#include "real_to_complex_impl.cuh"

template <typename T>
__global__ void ToComplex(const size_t size, const T *input, T *output, cudaStream_t cuda_stream) {
  // set the complex real to original real, imag to 0j
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[2 * pos] = input[pos];
  }
}

template <typename T>
cudaError_t RealToComplex(const size_t size, const T *input, T *output, cudaStream_t cuda_stream) {
  cudaMemsetAsync(output, 0, 2 * size * sizeof(T), cuda_stream);
  ToComplex<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output, cuda_stream);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t RealToComplex<double>(const size_t size, const double *input, double *output,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t RealToComplex<float>(const size_t size, const float *input, float *output,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t RealToComplex<int>(const size_t size, const int *input, int *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t RealToComplex<int64_t>(const size_t size, const int64_t *input, int64_t *output,
                                                            cudaStream_t cuda_stream);
