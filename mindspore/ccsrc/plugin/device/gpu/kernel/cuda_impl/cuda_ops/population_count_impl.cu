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
#include "population_count_impl.cuh"

template <typename T>
__global__ void PopulationCount(const size_t size, const T *input, uint8_t *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    output[i] = __popc(__ldg(input + i));
  }
  return;
}

template <>
__global__ void PopulationCount(const size_t size, const int8_t *input, uint8_t *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    output[i] = __popc(static_cast<uint8_t>(__ldg(input + i)));
  }
  return;
}

template <>
__global__ void PopulationCount(const size_t size, const int16_t *input, uint8_t *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    output[i] = __popc(static_cast<uint16_t>(__ldg(input + i)));
  }
  return;
}

template <>
__global__ void PopulationCount(const size_t size, const int64_t *input, uint8_t *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    output[i] = __popcll(__ldg(input + i));
  }
  return;
}

template <typename T>
cudaError_t CalPopulationCount(const size_t size, const T *input, uint8_t *output, const uint32_t &device_id,
                               cudaStream_t cuda_stream) {
  PopulationCount<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<uint8_t>(const size_t size, const uint8_t *input,
                                                                 uint8_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<int8_t>(const size_t size, const int8_t *input, uint8_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<uint16_t>(const size_t size, const uint16_t *input,
                                                                  uint8_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<int16_t>(const size_t size, const int16_t *input,
                                                                 uint8_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<uint32_t>(const size_t size, const uint32_t *input,
                                                                  uint8_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<int32_t>(const size_t size, const int32_t *input,
                                                                 uint8_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<uint64_t>(const size_t size, const uint64_t *input,
                                                                  uint8_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPopulationCount<int64_t>(const size_t size, const int64_t *input,
                                                                 uint8_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
