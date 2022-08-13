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
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_reshape_impl.cuh"
#include "include/cuda_runtime.h"

__global__ void CalSparseReshapeKernel(const int64_t *indices, const int64_t *shape, int64_t *y_indices,
                                       const int64_t *y_shape, const size_t indice_number_,
                                       const size_t shape_elements_, const size_t new_shape_elements_) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < indice_number_; pos += blockDim.x * gridDim.x) {
    const int64_t *input_index = indices + (pos * shape_elements_);
    int64_t *output_index = y_indices + (pos * new_shape_elements_);
    int64_t dense_index = 0;
    for (size_t i = 0; i < shape_elements_; ++i) {
      dense_index = dense_index * *(shape + i) + *(input_index + i);
    }
    for (int i = new_shape_elements_ - 1; i >= 0; --i) {
      int64_t output_size = *(y_shape + i);
      *(output_index + i) = dense_index % output_size;
      dense_index /= output_size;
    }
  }
}

void CalSparseReshape(const int64_t *indices, const int64_t *shape, int64_t *y_indices, int64_t *y_shape,
                      const size_t indice_number_, const size_t shape_elements_, const size_t new_shape_elements_,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalSparseReshapeKernel<<<CUDA_BLOCKS(device_id, indice_number_), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    indices, shape, y_indices, y_shape, indice_number_, shape_elements_, new_shape_elements_);
}
