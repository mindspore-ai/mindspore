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
                              const T *out_indices, size_t out_size, S *dx1, S *dx2, size_t dim, S init_val) {
    size_t stride = gridDim.x * blockDim.x;
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x1_idx = threadId;
    while (x1_idx < x1_size) {
      size_t idx = x1_idx * dim;
      auto x_idx = x1_indices[idx];
      auto y_idx = x1_indices[idx + 1];
      size_t catch_x1_i = 0;
      for (size_t j = 0; j < x1_size; j++) {
        auto oj = j * dim;
        if (x1_indices[oj] == x_idx && x1_indices[oj + 1] == y_idx) {
          if (x1_idx == j) {
            break;
          } else {
            catch_x1_i += 1;
          }
        }
      }
      S val = init_val;
      size_t same_x1_i = 0;
      for (size_t i = 0; i < out_size; i++) {
        auto oi = i * dim;
        if (out_indices[oi] == x_idx && out_indices[oi + 1] == y_idx) {
          if (same_x1_i == catch_x1_i) {
            val = dout[i];
            break;
          } else {
            same_x1_i += 1;
          }
        }
      }
      dx1[x1_idx] = val;
      x1_idx += stride;
    }

    size_t x2_idx = threadId;
    while (x2_idx < x2_size) {
      size_t idx = x2_idx * dim;
      auto x_idx = x2_indices[idx];
      auto y_idx = x2_indices[idx + 1];
      size_t catch_x2_i = 0;
      for (size_t j = 0; j < x2_size; j++) {
        auto oj = j * dim;
        if (x2_indices[oj] == x_idx && x2_indices[oj + 1] == y_idx) {
          if (x2_idx == j) {
            break;
          } else {
            catch_x2_i += 1;
          }
        }
      }
      S val = init_val;
      size_t same_x2_i = 0;
      for (size_t i = 0; i < out_size; i++) {
        auto oi = i * dim;
        if (out_indices[oi] == x_idx && out_indices[oi + 1] == y_idx) {
          if (same_x2_i == catch_x2_i) {
            val = dout[i];
            break;
          } else {
            same_x2_i += 1;
          }
        }
      }
      dx2[x2_idx] = val;
      x2_idx += stride;
    }
  return;
}

template <typename T, typename S>
void CalSparseAddGrad(const S *dout, const T *x1_indices, size_t x1_size, const T *x2_indices, size_t x2_size,
                      const T *out_indices, size_t out_size, S *dx1, S *dx2, size_t dim,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t max_in_size = std::max(x1_size, x2_size);
  SparseAddGrad<<<CUDA_BLOCKS(device_id, max_in_size), CUDA_THREADS(device_id), 0,
                  cuda_stream>>>(dout, x1_indices, x1_size, x2_indices, x2_size, out_indices, out_size,
                                 dx1, dx2, dim, S(0));
  return;
}

template<typename T>
void CalSparseAddGrad(const cuComplex *dout, const T *x1_indices, size_t x1_size, const T *x2_indices, size_t x2_size,
                      const T *out_indices, size_t out_size, cuComplex *dx1, cuComplex *dx2, size_t dim,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t max_in_size = std::max(x1_size, x2_size);
  SparseAddGrad<<<CUDA_BLOCKS(device_id, max_in_size), CUDA_THREADS(device_id), 0,
                  cuda_stream>>>(dout, x1_indices, x1_size, x2_indices, x2_size, out_indices, out_size,
                                 dx1, dx2, dim, {0, 0});
  return;
}

template<typename T>
void CalSparseAddGrad(const cuDoubleComplex *dout, const T *x1_indices, size_t x1_size, const T *x2_indices,
                      size_t x2_size, const T *out_indices, size_t out_size, cuDoubleComplex *dx1,
                      cuDoubleComplex *dx2, size_t dim, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  size_t max_in_size = std::max(x1_size, x2_size);
  SparseAddGrad<<<CUDA_BLOCKS(device_id, max_in_size), CUDA_THREADS(device_id), 0,
                  cuda_stream>>>(dout, x1_indices, x1_size, x2_indices, x2_size, out_indices, out_size,
                                 dx1, dx2, dim, {0, 0});
  return;
}

#define GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(index_type, val_type)                                    \
  template CUDA_LIB_EXPORT void CalSparseAddGrad<index_type, val_type>(const val_type *dout,         \
    const index_type *x1_indices, size_t x1_size, const index_type *x2_indices, size_t x2_size,      \
    const index_type *out_indices, size_t out_size, val_type *dx1, val_type *dx2, size_t dim,        \
    const uint32_t &device_id, cudaStream_t cuda_stream);

#define GPU_SPARSE_ADD_GRAD_COMPLEX_EXPORT_REGISTER(index_type, val_type)                            \
  template CUDA_LIB_EXPORT void CalSparseAddGrad<index_type>(const val_type *dout,                   \
    const index_type *x1_indices, size_t x1_size, const index_type *x2_indices, size_t x2_size,      \
    const index_type *out_indices, size_t out_size, val_type *dx1, val_type *dx2, size_t dim,        \
    const uint32_t &device_id, cudaStream_t cuda_stream);

GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int8_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int16_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int32_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int64_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, float)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, double)
GPU_SPARSE_ADD_GRAD_COMPLEX_EXPORT_REGISTER(int64_t, cuComplex)
GPU_SPARSE_ADD_GRAD_COMPLEX_EXPORT_REGISTER(int64_t, cuDoubleComplex)
