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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/compare_and_bitpack_impl.cuh"
#include <limits>


template <typename T>
__global__ void CompareAndBitpack(const T *x, const T *threshold, uint8_t *output, const size_t output_num) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    uint8_t res;
    res = (x[8 * pos] > *threshold) << 7;
    res = res | ((x[8 * pos + 1] > *threshold) << 6);
    res = res | ((x[8 * pos + 2] > *threshold) << 5);
    res = res | ((x[8 * pos + 3] > *threshold) << 4);
    res = res | ((x[8 * pos + 4] > *threshold) << 3);
    res = res | ((x[8 * pos + 5] > *threshold) << 2);
    res = res | ((x[8 * pos + 6] > *threshold) << 1);
    res = res | (x[8 * pos + 7] > *threshold);
    output[pos] = res;
  }
  return;
}

template <>
__global__ void CompareAndBitpack<bool>(const bool *x, const bool *threshold,
                                        uint8_t *output, const size_t output_num) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    uint8_t res;
    res = x[8 * pos] << 7;
    res = res | (x[8 * pos + 1] << 6);
    res = res | (x[8 * pos + 2] << 5);
    res = res | (x[8 * pos + 3] << 4);
    res = res | (x[8 * pos + 4] << 3);
    res = res | (x[8 * pos + 5] << 2);
    res = res | (x[8 * pos + 6] << 1);
    res = res | x[8 * pos + 7];
    output[pos] = res;
  }
  return;
}

template <typename T>
void CalCompareAndBitpack(const T *x, const T *threshold, uint8_t *output, const size_t output_num,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  CompareAndBitpack<<<CUDA_BLOCKS(device_id, output_num), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    x, threshold, output, output_num);
  return;
}

template CUDA_LIB_EXPORT void CalCompareAndBitpack<half>(
  const half *x, const half *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCompareAndBitpack<float>(
  const float *x, const float *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCompareAndBitpack<double>(
  const double *x, const double *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCompareAndBitpack<int8_t>(
  const int8_t *x, const int8_t *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCompareAndBitpack<int16_t>(
  const int16_t *x, const int16_t *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCompareAndBitpack<int32_t>(
  const int32_t *x, const int32_t *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCompareAndBitpack<int64_t>(
  const int64_t *x, const int64_t *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCompareAndBitpack<bool>(
  const bool *x, const bool *threshold, uint8_t *output, const size_t output_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
