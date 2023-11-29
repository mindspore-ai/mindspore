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

#include "bucketize_impl.cuh"

template <typename T>
__global__ void Bucketize(const int N, const int M, const float *bounds, const T *input, int32_t *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    int32_t low = -1;
    int32_t high = M;
    while (high - low > 1) {
      const int32_t median = low + (high - low) / 2;
      if (bounds[median] > static_cast<float>(input[i])) {
        high = median;
      } else {
        low = median;
      }
    }
    output[i] = high;
  }
  return;
}

template <typename T>
cudaError_t CalBucketize(const int size, const int M, const float *bounds, const T *input, int32_t *output,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  Bucketize<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, M, bounds, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBucketize<int32_t>(const int size, const int M, const float *bounds,
                                                           const int32_t *input, int32_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBucketize<int64_t>(const int size, const int M, const float *bounds,
                                                           const int64_t *input, int32_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBucketize<float>(const int size, const int M, const float *bounds,
                                                         const float *input, int32_t *output, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBucketize<double>(const int size, const int M, const float *bounds,
                                                          const double *input, int32_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
