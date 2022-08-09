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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/histogram_fixed_width_impl.cuh"
#include <cub/cub.cuh>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"

template <typename T>
void HistogramFixedWidthKernel(int num_samples, const T *d_samples, const double *d_levels,
                               int32_t *d_histogram, int64_t num_levels, cudaStream_t cuda_stream) {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  (void)cub::DeviceHistogram::HistogramRange(nullptr, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels,
                                             num_samples, cuda_stream);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
                                             d_levels, num_samples, cuda_stream);
  (void)cudaFree(d_temp_storage);
  return;
}

template <typename T>
void CalHistogramFixedWidth(int num_samples, const T *d_samples, const double *d_levels, int32_t *d_histogram,
                            int64_t num_levels, cudaStream_t cuda_stream) {
  HistogramFixedWidthKernel(num_samples, d_samples, d_levels, d_histogram, num_levels, cuda_stream);
  return;
}

template CUDA_LIB_EXPORT void CalHistogramFixedWidth<int32_t>(int num_samples, const int32_t *d_samples,
                                                              const double *d_levels, int32_t *d_histogram,
                                                              int64_t num_levels, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalHistogramFixedWidth<double>(int num_samples, const double *d_samples,
                                                             const double *d_levels, int32_t *d_histogram,
                                                             int64_t num_levels, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalHistogramFixedWidth<float>(int num_samples, const float *d_samples,
                                                            const double *d_levels, int32_t *d_histogram,
                                                            int64_t num_levels, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalHistogramFixedWidth<half>(int num_samples, const half *d_samples,
                                                           const double *d_levels, int32_t *d_histogram,
                                                           int64_t num_levels, cudaStream_t cuda_stream);
