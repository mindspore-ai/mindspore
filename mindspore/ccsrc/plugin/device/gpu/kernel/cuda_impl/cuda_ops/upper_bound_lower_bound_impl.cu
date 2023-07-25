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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upper_bound_lower_bound_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T, typename S>
__global__ void CalUpperBoundKernel(const size_t size, const size_t sorted_x_col_, const size_t values_col_,
                                    const T *sorted_x, const T *values, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t values_row = pos / values_col_;
    int64_t low = values_row * sorted_x_col_;
    int64_t up = (values_row + 1) * sorted_x_col_ - 1;
    while (low <= up) {
      int64_t mid = (low + up) / 2;
      if (values[pos] < sorted_x[static_cast<size_t>(mid)]) {
        up = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    output[pos] = static_cast<S>(low - values_row * sorted_x_col_);
  }
}

template <typename T, typename S>
__global__ void CalLowerBoundKernel(const size_t size, const size_t sorted_x_col_, const size_t values_col_,
                                    const T *sorted_x, const T *values, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t values_row = pos / values_col_;
    int64_t low = values_row * sorted_x_col_;
    int64_t up = (values_row + 1) * sorted_x_col_ - 1;
    while (low <= up) {
      int64_t mid = (low + up) / 2;
      if (values[pos] <= sorted_x[static_cast<size_t>(mid)]) {
        up = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    output[pos] = static_cast<S>(low - values_row * sorted_x_col_);
  }
}

template <typename T, typename S>
cudaError_t CalUpperBound(const size_t size, const size_t sorted_x_col_, const size_t values_col_, const T *sorted_x,
                          const T *values, S *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalUpperBoundKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, sorted_x_col_, values_col_, sorted_x, values, output);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t CalLowerBound(const size_t size, const size_t sorted_x_col_, const size_t values_col_, const T *sorted_x,
                          const T *values, S *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalLowerBoundKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, sorted_x_col_, values_col_, sorted_x, values, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int8_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                const size_t values_col_, const int8_t *sorted_x,
                                                                const int8_t *values, int *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int16_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                 const size_t values_col_, const int16_t *sorted_x,
                                                                 const int16_t *values, int *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int32_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                 const size_t values_col_, const int32_t *sorted_x,
                                                                 const int32_t *values, int *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int64_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                 const size_t values_col_, const int64_t *sorted_x,
                                                                 const int64_t *values, int *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<uint16_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                  const size_t values_col_, const uint16_t *sorted_x,
                                                                  const uint16_t *values, int *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<half, int>(const size_t size, const size_t sorted_x_col_,
                                                              const size_t values_col_, const half *sorted_x,
                                                              const half *values, int *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<float, int>(const size_t size, const size_t sorted_x_col_,
                                                               const size_t values_col_, const float *sorted_x,
                                                               const float *values, int *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<double, int>(const size_t size, const size_t sorted_x_col_,
                                                                const size_t values_col_, const double *sorted_x,
                                                                const double *values, int *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int8_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                    const size_t values_col_, const int8_t *sorted_x,
                                                                    const int8_t *values, int64_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int16_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                     const size_t values_col_, const int16_t *sorted_x,
                                                                     const int16_t *values, int64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int32_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                     const size_t values_col_, const int32_t *sorted_x,
                                                                     const int32_t *values, int64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<int64_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                     const size_t values_col_, const int64_t *sorted_x,
                                                                     const int64_t *values, int64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<uint16_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                      const size_t values_col_,
                                                                      const uint16_t *sorted_x, const uint16_t *values,
                                                                      int64_t *output, const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<half, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                  const size_t values_col_, const half *sorted_x,
                                                                  const half *values, int64_t *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<float, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                   const size_t values_col_, const float *sorted_x,
                                                                   const float *values, int64_t *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUpperBound<double, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                    const size_t values_col_, const double *sorted_x,
                                                                    const double *values, int64_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int8_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                const size_t values_col_, const int8_t *sorted_x,
                                                                const int8_t *values, int *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int16_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                 const size_t values_col_, const int16_t *sorted_x,
                                                                 const int16_t *values, int *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int32_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                 const size_t values_col_, const int32_t *sorted_x,
                                                                 const int32_t *values, int *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int64_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                 const size_t values_col_, const int64_t *sorted_x,
                                                                 const int64_t *values, int *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<uint16_t, int>(const size_t size, const size_t sorted_x_col_,
                                                                  const size_t values_col_, const uint16_t *sorted_x,
                                                                  const uint16_t *values, int *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<half, int>(const size_t size, const size_t sorted_x_col_,
                                                              const size_t values_col_, const half *sorted_x,
                                                              const half *values, int *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<float, int>(const size_t size, const size_t sorted_x_col_,
                                                               const size_t values_col_, const float *sorted_x,
                                                               const float *values, int *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<double, int>(const size_t size, const size_t sorted_x_col_,
                                                                const size_t values_col_, const double *sorted_x,
                                                                const double *values, int *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int8_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                    const size_t values_col_, const int8_t *sorted_x,
                                                                    const int8_t *values, int64_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int16_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                     const size_t values_col_, const int16_t *sorted_x,
                                                                     const int16_t *values, int64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int32_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                     const size_t values_col_, const int32_t *sorted_x,
                                                                     const int32_t *values, int64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<int64_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                     const size_t values_col_, const int64_t *sorted_x,
                                                                     const int64_t *values, int64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<uint16_t, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                      const size_t values_col_,
                                                                      const uint16_t *sorted_x, const uint16_t *values,
                                                                      int64_t *output, const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<half, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                  const size_t values_col_, const half *sorted_x,
                                                                  const half *values, int64_t *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<float, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                   const size_t values_col_, const float *sorted_x,
                                                                   const float *values, int64_t *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLowerBound<double, int64_t>(const size_t size, const size_t sorted_x_col_,
                                                                    const size_t values_col_, const double *sorted_x,
                                                                    const double *values, int64_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
