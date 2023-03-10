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

#include "approximate_equal_impl.cuh"

__inline__ __device__ float HalfFloatInputConvert(const half val) { return __half2float(val); }
__inline__ __device__ float HalfFloatInputConvert(const float val) { return val; }
__inline__ __device__ double HalfFloatInputConvert(const double val) { return val; }

template <typename T>
__global__ void ApproximateEqual(const size_t size, const T *input_x1, const T *input_x2, const float tolerance,
                                 bool *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = std::abs(HalfFloatInputConvert(input_x1[pos] - input_x2[pos])) < tolerance ? true : false;
  }
}

template <typename T>
cudaError_t CalApproximateEqual(const size_t size, const T *input_x1, const T *input_x2, const float tolerance,
                                bool *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  ApproximateEqual<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input_x1, input_x2,
                                                                                              tolerance, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalApproximateEqual<half>(const size_t size, const half *input_x1,
                                                               const half *input_x2, const float tolerance,
                                                               bool *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalApproximateEqual<float>(const size_t size, const float *input_x1,
                                                                const float *input_x2, const float tolerance,
                                                                bool *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalApproximateEqual<double>(const size_t size, const double *input_x1,
                                                                 const double *input_x2, const float tolerance,
                                                                 bool *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
