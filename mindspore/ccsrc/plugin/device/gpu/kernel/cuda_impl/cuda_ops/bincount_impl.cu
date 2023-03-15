/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

// #include <algorithm>
#include <cub/cub.cuh>
#include "bincount_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
void BincountNoWeight(const int32_t *array, const int32_t *size, T *bins, const int64_t threads_size,
                      const int64_t outer_size) {
  const int32_t *d_samples = array;
  T *d_histogram = bins;
  int32_t num_levels = outer_size + 1;
  int32_t lower_level = int32_t(0);
  int32_t upper_level = outer_size;
  int32_t num_samples = threads_size;

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  // HistogramEven no support int64_t and double
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
                                      lower_level, upper_level, num_samples);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
                                      lower_level, upper_level, num_samples);
  (void)cudaFree(d_temp_storage);
}

template <>
void BincountNoWeight(const int32_t *array, const int32_t *size, int64_t *bins, const int64_t threads_size,
                      const int64_t outer_size) {}

template <>
void BincountNoWeight(const int32_t *array, const int32_t *size, double *bins, const int64_t threads_size,
                      const int64_t outer_size) {}

template <typename T>
__global__ void Bincount(const int32_t *array, const int32_t *size, const T *weight, T *bins,
                         const int64_t threads_size, const bool has_weights) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < threads_size; pos += blockDim.x * gridDim.x) {
    int32_t index = array[pos];
    if (index >= 0 && index < size[0]) {
      MsAtomicAdd(bins + index, has_weights ? weight[pos] : T(1));
    }
  }
  return;
}

template <typename T>
__global__ void BincountMem(const int32_t *array, const int32_t *size, const T *weight, T *bins,
                            const int64_t threads_size, const int64_t outer_size, const bool has_weights) {
  const int BLOCK_DIM = 6 * 1024;
  __shared__ T smem[BLOCK_DIM];
  for (size_t index = threadIdx.x; index < outer_size; index += blockDim.x) {
    smem[index] = T(0);
  }
  __syncthreads();
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < threads_size; pos += blockDim.x * gridDim.x) {
    int32_t index = array[pos];
    if (index >= 0 && index < size[0]) {
      MsAtomicAdd(smem + index, has_weights ? weight[pos] : T(1));
    }
  }
  __syncthreads();
  for (size_t index = threadIdx.x; index < outer_size; index += blockDim.x) {
    MsAtomicAdd(bins + index, smem[index]);
  }
  return;
}

template <typename T>
void CalBincount(const int32_t *array, const int32_t *size, const T *weight, T *bins, const bool has_weights,
                 const int64_t threads_size, const int64_t outer_size, const uint32_t &device_id,
                 cudaStream_t cuda_stream) {
  cudaMemsetAsync(bins, 0, sizeof(T) * outer_size);
  if (!has_weights) {
    BincountNoWeight(array, size, bins, threads_size, outer_size);
    return;
  }
  if (outer_size <= 6 * 1024) {
    BincountMem<<<CUDA_BLOCKS(device_id, threads_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      array, size, weight, bins, threads_size, outer_size, has_weights);
  } else {
    Bincount<<<CUDA_BLOCKS(device_id, threads_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      array, size, weight, bins, threads_size, has_weights);
  }
  return;
}

template <>
void CalBincount(const int32_t *array, const int32_t *size, const double *weight, double *bins, const bool has_weights,
                 const int64_t threads_size, const int64_t outer_size, const uint32_t &device_id,
                 cudaStream_t cuda_stream) {
  cudaMemsetAsync(bins, 0, sizeof(double) * outer_size);
  if (outer_size <= 6 * 1024) {
    BincountMem<<<CUDA_BLOCKS(device_id, threads_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      array, size, weight, bins, threads_size, outer_size, has_weights);
  } else {
    Bincount<<<CUDA_BLOCKS(device_id, threads_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      array, size, weight, bins, threads_size, has_weights);
  }
  return;
}

template <>
void CalBincount(const int32_t *array, const int32_t *size, const int64_t *weight, int64_t *bins,
                 const bool has_weights, const int64_t threads_size, const int64_t outer_size,
                 const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaMemsetAsync(bins, 0, sizeof(int64_t) * outer_size);
  if (outer_size <= 6 * 1024) {
    BincountMem<<<CUDA_BLOCKS(device_id, threads_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      array, size, weight, bins, threads_size, outer_size, has_weights);
  } else {
    Bincount<<<CUDA_BLOCKS(device_id, threads_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      array, size, weight, bins, threads_size, has_weights);
  }
  return;
}

template CUDA_LIB_EXPORT void CalBincount<float>(const int32_t *array, const int32_t *size, const float *weight,
                                                 float *bins, const bool has_weights, const int64_t threads_size,
                                                 const int64_t outer_size, const uint32_t &device_id,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBincount<double>(const int32_t *array, const int32_t *size, const double *weight,
                                                  double *bins, const bool has_weights, const int64_t threads_size,
                                                  const int64_t outer_size, const uint32_t &device_id,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBincount<int32_t>(const int32_t *array, const int32_t *size, const int32_t *weight,
                                                   int32_t *bins, const bool has_weights, const int64_t threads_size,
                                                   const int64_t outer_size, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBincount<int64_t>(const int32_t *array, const int32_t *size, const int64_t *weight,
                                                   int64_t *bins, const bool has_weights, const int64_t threads_size,
                                                   const int64_t outer_size, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
