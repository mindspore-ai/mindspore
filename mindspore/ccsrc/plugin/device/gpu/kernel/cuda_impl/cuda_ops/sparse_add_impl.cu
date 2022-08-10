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

#include <stdint.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_add_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

constexpr size_t kNumOfColumn = 2;

template <typename T>
__device__ int CompareTwoIndices(const T *a_indices, const T *b_indices, size_t a_row, size_t b_row,
                                 const size_t dims) {
  for (size_t dim = 0; dim < dims; dim++) {
    auto a_idx = a_indices[a_row * kNumOfColumn + dim];
    auto b_idx = b_indices[b_row * kNumOfColumn + dim];
    if (a_idx < b_idx) {
      return -1;
    } else if (a_idx > b_idx) {
      return 1;
    }
  }
  return 0;
}

template <typename S>
__device__ void InitValue(S *whole_values, size_t index) {
  whole_values[index] = 0;
}

template <>
__device__ void InitValue<cuComplex>(cuComplex *whole_values, size_t index) {
  whole_values[index] = {0, 0};
}

template <>
__device__ void InitValue<cuDoubleComplex>(cuDoubleComplex *whole_values, size_t index) {
  whole_values[index] = {0, 0};
}


template <typename T, typename S>
__device__ void SparseAddPreprocess(const T *a_indices, const S *a_values, const T *b_indices, const S *b_values,
                      size_t *a_value_index, size_t *b_value_index, bool *is_from_a, S* whole_values,
                      size_t *place_holder_index, int64_t *indices, bool *threshold_valid,
                      size_t *i, size_t *j, size_t *full_count, size_t *cross_count,
                      const size_t a_indices_num, const size_t b_indices_num) {
  while (*i < a_indices_num && *j < b_indices_num) {
    switch (CompareTwoIndices(a_indices, b_indices, *i, *j, kNumOfColumn)) {
      case -1:
        is_from_a[*full_count] = true;
        indices[*full_count] = *i;
        whole_values[*full_count] = a_values[*i];
        threshold_valid[*full_count] = true;
        *i += 1;
        *full_count += 1;
        break;
      case 0:
        is_from_a[*full_count] = true;
        indices[*full_count] = *i;
        a_value_index[*cross_count] = *i;
        b_value_index[*cross_count] = *j;
        place_holder_index[*cross_count] = *full_count;
        InitValue(whole_values, *full_count);
        threshold_valid[*full_count] = true;
        *i += 1;
        *j += 1;
        *full_count += 1;
        *cross_count += 1;
        break;
      case 1:
        is_from_a[*full_count] = false;
        indices[*full_count] = *j;
        whole_values[*full_count] = b_values[*j];
        threshold_valid[*full_count] = true;
        *j += 1;
        *full_count += 1;
        break;
    }
  }
}

template <typename S, typename K>
__device__ bool IsInLimit(S val, K limit) {
  return limit > std::abs(val);
}

template <typename K>
__device__ bool IsInLimit(cuComplex val, K limit) {
  return (limit * limit) > (val.x * val.x + val.y * val.y);
}

template <typename K>
__device__ bool IsInLimit(cuDoubleComplex val, K limit) {
  return (limit * limit) > (val.x * val.x + val.y * val.y);
}

template <typename T, typename S, typename K>
__device__ void SparseAddPostprocess(const T *a_indices, const S *a_values, const T *b_indices, const S *b_values,
                                     const S* res_store_mem, T *sum_indices, S *sum_values,
                                     size_t *a_value_index, size_t *b_value_index, bool *is_from_a, S* whole_values,
                                     size_t *place_holder_index, int64_t *indices, bool *threshold_valid,
                                     size_t *i, size_t *j, size_t *full_count, size_t *cross_count, int64_t *sum_count,
                                     const size_t a_indices_num, const size_t b_indices_num, K threshold) {
  for (size_t calculate_num = 0; calculate_num < *cross_count; calculate_num++) {
    if (IsInLimit(res_store_mem[calculate_num], *threshold)) {
      threshold_valid[place_holder_index[calculate_num]] = false;
    } else {
      whole_values[place_holder_index[calculate_num]] = res_store_mem[calculate_num];
    }
  }

  if (*i < a_indices_num) {
    while (*i < a_indices_num) {
      indices[*full_count] = *i;
      is_from_a[*full_count] = true;
      threshold_valid[*full_count] = true;
      whole_values[*full_count] = a_values[*i];
      *i += 1;
      *full_count += 1;
    }
  } else {
    while (*j < b_indices_num) {
      indices[*full_count] = *j;
      is_from_a[*full_count] = false;
      threshold_valid[*full_count] = true;
      whole_values[*full_count] = b_values[*j];
      *j += 1;
      *full_count += 1;
    }
  }

  size_t offset = 0;
  for (size_t num = 0; num < *full_count; num++) {
    bool copy_from_a = is_from_a[num];
    int64_t index_from_input = indices[num];
    if (!threshold_valid[num]) {
      offset += 1;
      continue;
    } else {
      if (copy_from_a) {
        for (size_t column = 0; column < kNumOfColumn; column++) {
          sum_indices[(num - offset) * kNumOfColumn + column] = a_indices[index_from_input * kNumOfColumn + column];
        }
      } else {
        for (size_t column = 0; column < kNumOfColumn; column++) {
          sum_indices[(num - offset) * kNumOfColumn + column] = b_indices[index_from_input * kNumOfColumn + column];
        }
      }
    }
    sum_values[num - offset] = whole_values[num];
    *sum_count += 1;
  }
}

template <typename S>
__device__ S AddImpl(S a, S b) {
  return a + b;
}

template <>
__device__ cuComplex AddImpl(cuComplex a, cuComplex b) {
  return cuCaddf(a, b);
}

template <>
__device__ cuDoubleComplex AddImpl(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCadd(a, b);
}

template <typename T, typename S, typename K>
__global__ void SparseAddKernel(const T *a_indices, const S *a_values, const T *b_indices, const S *b_values,
                      T *sum_indices, S *sum_values,
                      size_t *a_value_index, size_t *b_value_index, bool *is_from_a, S* whole_values,
                      size_t *place_holder_index, int64_t *indices, bool *threshold_valid,
                      const size_t a_indices_num, const size_t b_indices_num,
                      S *res_store_mem, int64_t *sum_count,
                      const K* threshold) {
  size_t i = 0, j = 0;
  size_t full_count = 0;
  size_t cross_count = 0;

  SparseAddPreprocess(a_indices, a_values, b_indices, b_values,
                      a_value_index, b_value_index, is_from_a, whole_values,
                      place_holder_index, indices, threshold_valid,
                      &i, &j, &full_count, &cross_count,
                      a_indices_num, b_indices_num);

  for (size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < cross_count; k += blockDim.x * gridDim.x) {
    res_store_mem[k] = AddImpl(a_values[a_value_index[k]], b_values[b_value_index[k]]);
  }

  SparseAddPostprocess(a_indices, a_values, b_indices, b_values,
                      res_store_mem, sum_indices, sum_values,
                      a_value_index, b_value_index, is_from_a, whole_values,
                      place_holder_index, indices, threshold_valid,
                      &i, &j, &full_count, &cross_count, sum_count,
                      a_indices_num, b_indices_num, threshold);
}

template <typename T, typename S, typename K>
void SparseAdd(const T *a_indices, const S *a_values, const T *b_indices, const S *b_values,
               T *sum_indices, S *sum_values,
               size_t *a_value_index, size_t *b_value_index, bool *is_from_a, S* whole_values,
               size_t *place_holder_index, int64_t *indices, bool *threshold_valid,
               const size_t a_indices_num, const size_t b_indices_num,
               S *res_store_mem, int64_t *sum_count,
               const K *threshold,  const uint32_t &device_id,
               cudaStream_t cuda_stream) {
  SparseAddKernel<<<GET_BLOCKS(1), 1, 0, cuda_stream>>>(
                      a_indices, a_values, b_indices, b_values,
                      sum_indices, sum_values,
                      a_value_index, b_value_index, is_from_a, whole_values,
                      place_holder_index, indices, threshold_valid,
                      a_indices_num, b_indices_num,
                      res_store_mem, sum_count, threshold);
}

#define GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(index_type, val_type, thr_type)                      \
  template CUDA_LIB_EXPORT void SparseAdd<index_type, val_type, thr_type>(const index_type *a_indices,   \
    const val_type *a_values, const index_type *b_indices, const val_type *b_values,           \
    index_type *sum_indices, val_type *sum_values,                                             \
    size_t *a_value_index, size_t *b_value_index, bool *is_from_a, val_type* whole_values,     \
    size_t *place_holder_index, int64_t *indices, bool *threshold_valid,                       \
    const size_t a_indices_num, const size_t b_indices_num,                                    \
    val_type *res_store_mem, int64_t *sum_count,                                               \
    const thr_type* threshold,  const uint32_t &device_id,                                     \
    cudaStream_t cuda_stream);

GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int8_t, int8_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int16_t, int16_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int32_t, int32_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, int64_t, int64_t)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, float, float)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, double, double)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, cuComplex, float)
GPU_SPARSE_ADD_GRAD_EXPORT_REGISTER(int64_t, cuDoubleComplex, double)
