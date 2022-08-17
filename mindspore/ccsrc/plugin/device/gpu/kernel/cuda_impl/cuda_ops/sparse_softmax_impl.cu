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

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_softmax_impl.cuh"

template <typename T>
__global__ void CalSparseSoftmaxKernel(const int64_t *indices, const T *values, T *output, int32_t *reorder,
                                       const size_t indice_dims, const size_t values_elements) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_elements; pos += blockDim.x * gridDim.x) {
    T out = std::exp(values[reorder[pos]]);
    T exp_sum = out;
    int64_t right_pos = pos + 1;
    int64_t left_pos = pos - 1;
    bool right_flag = true;
    bool left_flag = true;
    do {
      if (right_pos >= (int64_t)values_elements) {
        right_flag = false;
      } else {
        for (size_t i = 0; i + 1 < indice_dims; ++i) {
          if (indices[reorder[pos] * indice_dims + i] != indices[reorder[right_pos] * indice_dims + i]) {
            right_flag = false;
            break;
          }
        }
      }
      if (left_pos < 0) {
        left_flag = false;
      } else {
        for (size_t i = 0; i + 1 < indice_dims; ++i) {
          if (indices[reorder[pos] * indice_dims + i] != indices[reorder[left_pos] * indice_dims + i]) {
            left_flag = false;
            break;
          }
        }
      }
      if (right_flag) {
        exp_sum += std::exp(values[reorder[right_pos]]);
        right_pos++;
      }
      if (left_flag) {
        exp_sum += std::exp(values[reorder[left_pos]]);
        left_pos--;
      }
    } while (right_flag || left_flag);
    output[pos] = out / exp_sum;
  }
}

struct cmp_indices {
  const int64_t *indices_;
  const size_t indice_dims_;

  cmp_indices(const int64_t *ptr, const size_t dims)
    : indices_(ptr), indice_dims_(dims) {}

  __host__ __device__
  bool operator()(int64_t i, int64_t j) {
    for (size_t d = 0; d < indice_dims_; ++d) {
      if (indices_[i * indice_dims_ + d] < indices_[j * indice_dims_ + d]) {
        return true;
      }
      if (indices_[i * indice_dims_ + d] > indices_[j * indice_dims_ + d]) {
        break;
      }
    }
    return false;
  }
};

template <typename T>
void CalSparseSoftmax(const int64_t *indices, const T *values, T *output, int32_t *reorder,
                      int64_t *indice_to_num, const size_t indice_dims, const size_t values_elements,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::sequence(policy,
                   thrust::device_pointer_cast(reorder),
                   thrust::device_pointer_cast(reorder) + values_elements);
  thrust::sort(policy,
               thrust::device_pointer_cast(reorder),
               thrust::device_pointer_cast(reorder) + values_elements,
               cmp_indices(indices, indice_dims));
  CalSparseSoftmaxKernel<<<CUDA_BLOCKS(device_id, values_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    indices, values, output, reorder, indice_dims, values_elements);
}

template CUDA_LIB_EXPORT void CalSparseSoftmax<float>(const int64_t *indices, const float *values, float *output,
                                                      int32_t *reorder, int64_t *indice_to_num,
                                                      const size_t indice_dims, const size_t values_elements,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseSoftmax<double>(const int64_t *indices, const double *values, double *output,
                                                       int32_t *reorder, int64_t *indice_to_num,
                                                       const size_t indice_dims, const size_t values_elements,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
