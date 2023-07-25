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

#include "sparse_to_dense_impl.cuh"
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <limits>
#include <algorithm>
#include "include/cuda_fp16.h"

template <typename T>
__global__ void SetDefaultValue(const T default_value, const int64_t output_elements, T *output) {
  for (size_t ops = blockIdx.x * blockDim.x + threadIdx.x; ops < output_elements; ops += blockDim.x * gridDim.x) {
    output[ops] = default_value;
  }
}

template <typename T>
cudaError_t CallSetDefaultValue(const T default_value, const int64_t output_elements, T *output,
                                const uint32_t &device_id, cudaStream_t cuda_stream) {
  SetDefaultValue<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    default_value, output_elements, output);
  return GetCudaStatus();
}

template <typename T, typename Index>
__global__ void SparseToDense(const Index *indices, const T *vals, const int num_elems, const int num_vals,
                              const Index *output_shape, const int ndims, T *output) {
  for (size_t ops = blockIdx.x * blockDim.x + threadIdx.x; ops < num_elems; ops += blockDim.x * gridDim.x) {
    int64_t output_idx = indices[ops * ndims + ndims - 1];
    Index strides = 1;
    for (int i = ndims - 2; i >= 0; i--) {
      strides *= output_shape[i + 1];
      output_idx += indices[ops * ndims + i] * strides;
    }
    // If num_vals == 1, broadcast the scalar to the positions for non-zeros.
    output[output_idx] = vals[(num_vals == 1) ? 0 : ops];
  }
  __syncthreads();
}

template <typename T, typename Index>
cudaError_t CallSparseToDense(const Index *indices, const T *vals, const int num_elems, const int num_vals,
                              const Index *output_shape, const int ndims, T *output, const uint32_t &device_id,
                              cudaStream_t cuda_stream) {
  SparseToDense<<<CUDA_BLOCKS(device_id, num_elems), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    indices, vals, num_elems, num_vals, output_shape, ndims, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<bool>(bool default_value, const int64_t output_elements,
                                                               bool *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<int8_t>(int8_t default_value, const int64_t output_elements,
                                                                 int8_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<int16_t>(int16_t default_value, const int64_t output_elements,
                                                                  int16_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<int32_t>(int32_t default_value, const int64_t output_elements,
                                                                  int32_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<int64_t>(int64_t default_value, const int64_t output_elements,
                                                                  int64_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<uint8_t>(uint8_t default_value, const int64_t output_elements,
                                                                  uint8_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<uint16_t>(uint16_t default_value,
                                                                   const int64_t output_elements, uint16_t *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<half>(half default_value, const int64_t output_elements,
                                                               half *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<float>(float default_value, const int64_t output_elements,
                                                                float *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSetDefaultValue<double>(double default_value, const int64_t output_elements,
                                                                 double *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<bool, int32_t>(const int32_t *indices, const bool *vals,
                                                                      const int num_elems, const int num_vals,
                                                                      const int32_t *output_shape, const int ndims,
                                                                      bool *output, const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int8_t, int32_t>(const int32_t *indices, const int8_t *vals,
                                                                        const int num_elems, const int num_vals,
                                                                        const int32_t *output_shape, const int ndims,
                                                                        int8_t *output, const uint32_t &device_id,
                                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int16_t, int32_t>(const int32_t *indices, const int16_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int32_t *output_shape, const int ndims,
                                                                         int16_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int32_t, int32_t>(const int32_t *indices, const int32_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int32_t *output_shape, const int ndims,
                                                                         int32_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int64_t, int32_t>(const int32_t *indices, const int64_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int32_t *output_shape, const int ndims,
                                                                         int64_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<uint8_t, int32_t>(const int32_t *indices, const uint8_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int32_t *output_shape, const int ndims,
                                                                         uint8_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<uint16_t, int32_t>(const int32_t *indices, const uint16_t *vals,
                                                                          const int num_elems, const int num_vals,
                                                                          const int32_t *output_shape, const int ndims,
                                                                          uint16_t *output, const uint32_t &device_id,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<half, int32_t>(const int32_t *indices, const half *vals,
                                                                      const int num_elems, const int num_vals,
                                                                      const int32_t *output_shape, const int ndims,
                                                                      half *output, const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<float, int32_t>(const int32_t *indices, const float *vals,
                                                                       const int num_elems, const int num_vals,
                                                                       const int32_t *output_shape, const int ndims,
                                                                       float *output, const uint32_t &device_id,
                                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<double, int32_t>(const int32_t *indices, const double *vals,
                                                                        const int num_elems, const int num_vals,
                                                                        const int32_t *output_shape, const int ndims,
                                                                        double *output, const uint32_t &device_id,
                                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<bool, int64_t>(const int64_t *indices, const bool *vals,
                                                                      const int num_elems, const int num_vals,
                                                                      const int64_t *output_shape, const int ndims,
                                                                      bool *output, const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int8_t, int64_t>(const int64_t *indices, const int8_t *vals,
                                                                        const int num_elems, const int num_vals,
                                                                        const int64_t *output_shape, const int ndims,
                                                                        int8_t *output, const uint32_t &device_id,
                                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int16_t, int64_t>(const int64_t *indices, const int16_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int64_t *output_shape, const int ndims,
                                                                         int16_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int32_t, int64_t>(const int64_t *indices, const int32_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int64_t *output_shape, const int ndims,
                                                                         int32_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<int64_t, int64_t>(const int64_t *indices, const int64_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int64_t *output_shape, const int ndims,
                                                                         int64_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<uint8_t, int64_t>(const int64_t *indices, const uint8_t *vals,
                                                                         const int num_elems, const int num_vals,
                                                                         const int64_t *output_shape, const int ndims,
                                                                         uint8_t *output, const uint32_t &device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<uint16_t, int64_t>(const int64_t *indices, const uint16_t *vals,
                                                                          const int num_elems, const int num_vals,
                                                                          const int64_t *output_shape, const int ndims,
                                                                          uint16_t *output, const uint32_t &device_id,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<half, int64_t>(const int64_t *indices, const half *vals,
                                                                      const int num_elems, const int num_vals,
                                                                      const int64_t *output_shape, const int ndims,
                                                                      half *output, const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<float, int64_t>(const int64_t *indices, const float *vals,
                                                                       const int num_elems, const int num_vals,
                                                                       const int64_t *output_shape, const int ndims,
                                                                       float *output, const uint32_t &device_id,
                                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallSparseToDense<double, int64_t>(const int64_t *indices, const double *vals,
                                                                        const int num_elems, const int num_vals,
                                                                        const int64_t *output_shape, const int ndims,
                                                                        double *output, const uint32_t &device_id,
                                                                        cudaStream_t cuda_stream);
