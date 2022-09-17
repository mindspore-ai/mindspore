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

#include "fill_diagonal_impl.cuh"

template <typename T>
__global__ void FillDiagonal(const size_t size, const float fill_value, const int64_t step_size, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[step_size * pos] = static_cast<T>(fill_value);
  }
  return;
}


template <typename T>
void CalFillDiagonal(const size_t size, const float fill_value, const int64_t step_size, T *output,
                     const uint32_t &device_id, cudaStream_t cuda_stream) {
  FillDiagonal<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, fill_value, step_size,
                                                                                          output);
  return;
}

template
CUDA_LIB_EXPORT void CalFillDiagonal<float>(const size_t size, const float fill_value, const int64_t step_size,
                                            float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalFillDiagonal<int32_t>(const size_t size, const float fill_value, const int64_t step_size,
                                              int32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalFillDiagonal<int64_t>(const size_t size, const float fill_value, const int64_t step_size,
                                              int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

