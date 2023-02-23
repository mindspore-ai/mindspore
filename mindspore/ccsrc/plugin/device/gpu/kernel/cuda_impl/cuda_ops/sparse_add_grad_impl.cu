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

#include "sparse_add_grad_impl.cuh"
#include <algorithm>
template <typename T, typename S>
__global__ void SparseAddGrad(const S *dout, const T *x1_indices, size_t x1_size, const T *x2_indices, size_t x2_size,
                              const T *out_indices, size_t out_size, T *temp_save_ptr, S *dx1, S *dx2, size_t dim,
                              S init_val) {
  size_t stride = gridDim.x * blockDim.x;
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t x1_idx = threadId;
  memset(dx1, 0, sizeof(T) * x1_size);
  memset(dx2, 0, sizeof(T) * x2_size);
  while (x1_idx < x1_size) {
    size_t idx = x1_idx * dim;
    for (size_t i = 0; i < dim; i++) {
      temp_save_ptr[i] = x1_indices[idx + i];
    }
    for (size_t i = 0; i < out_size; i++) {
      auto oi = i * dim;
      bool same_flag = true;
      for (size_t j = 0; j < dim; j++) {
        if (temp_save_ptr[j] != out_indices[oi + j]) {
          same_flag = false;
          break;
        }
      }
      if (same_flag) {
        dx1[x1_idx] = dout[i];
        break;
      }
    }
    x1_idx += stride;
  }

  size_t x2_idx = threadId;
  while (x2_idx < x2_size) {
    size_t idx = x2_idx * dim;
    for (size_t i = 0; i < dim; i++) {
      temp_save_ptr[i] = x2_indices[idx + i];
    }
    for (size_t i = 0; i < out_size; i++) {
      auto oi = i * dim;
      bool same_flag = true;
      for (size_t j = 0; j < dim; j++) {
        if (temp_save_ptr[j] != out_indices[oi + j]) {
          same_flag = false;
          break;
        }
      }
      if (same_flag) {
        dx2[x2_idx] = dout[i];
        break;
      }
    }
    x2_idx += stride;
  }
  return;
}

template <typename T, typename S>
void CalSparseAddGrad(const S *dout, const T *x1_indices, size_t x1_size, const T *x2_indices, size_t x2_size,
                      const T *out_indices, size_t out_size, T *temp_save_ptr, S *dx1, S *dx2, size_t dim,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  dim3 blockSize(1);
  dim3 gridSize(1);
  SparseAddGrad<<<gridSize, blockSize, 0, cuda_stream>>>(dout, x1_indices, x1_size, x2_indices, x2_size, out_indices,
                                                         out_size, temp_save_ptr, dx1, dx2, dim, S(0));
  return;
}

template <typename T>
void CalSparseAddGrad(const cuComplex *dout, const T *x1_indices, size_t x1_size, const T *x2_indices, size_t x2_size,
                      const T *out_indices, size_t out_size, T *temp_save_ptr, cuComplex *dx1, cuComplex *dx2,
                      size_t dim, const uint32_t &device_id, cudaStream_t cuda_stream) {
  dim3 blockSize(1);
  dim3 gridSize(1);
  SparseAddGrad<<<gridSize, blockSize, 0, cuda_stream>>>(dout, x1_indices, x1_size, x2_indices, x2_size, out_indices,
                                                         out_size, temp_save_ptr, dx1, dx2, dim, {0, 0});
  return;
}

template <typename T>
void CalSparseAddGrad(const cuDoubleComplex *dout, const T *x1_indices, size_t x1_size, const T *x2_indices,
                      size_t x2_size, const T *out_indices, size_t out_size, T *temp_save_ptr, cuDoubleComplex *dx1,
                      cuDoubleComplex *dx2, size_t dim, const uint32_t &device_id, cudaStream_t cuda_stream) {
  dim3 blockSize(1);
  dim3 gridSize(1);
  SparseAddGrad<<<gridSize, blockSize, 0, cuda_stream>>>(dout, x1_indices, x1_size, x2_indices, x2_size, out_indices,
                                                         out_size, temp_save_ptr, dx1, dx2, dim, {0, 0});
  return;
}

#define GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(index_type, val_type)                                                     \
  template CUDA_LIB_EXPORT void CalSparseAddGrad<index_type, val_type>(                                               \
    const val_type *dout, const index_type *x1_indices, size_t x1_size, const index_type *x2_indices, size_t x2_size, \
    const index_type *out_indices, size_t out_size, index_type *temp_save_ptr, val_type *dx1, val_type *dx2,          \
    size_t dim, const uint32_t &device_id, cudaStream_t cuda_stream);

#define GPU_SPARSE_ADD_GRAD_COMPLEX_EXPORT_REGISTER(index_type, val_type)                                             \
  template CUDA_LIB_EXPORT void CalSparseAddGrad<index_type>(                                                         \
    const val_type *dout, const index_type *x1_indices, size_t x1_size, const index_type *x2_indices, size_t x2_size, \
    const index_type *out_indices, size_t out_size, index_type *temp_save_ptr, val_type *dx1, val_type *dx2,          \
    size_t dim, const uint32_t &device_id, cudaStream_t cuda_stream);

GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int8_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int16_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int32_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int64_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, float)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, double)
GPU_SPARSE_ADD_GRAD_COMPLEX_EXPORT_REGISTER(int64_t, cuComplex)
GPU_SPARSE_ADD_GRAD_COMPLEX_EXPORT_REGISTER(int64_t, cuDoubleComplex)
