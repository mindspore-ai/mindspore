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

#include <cub/cub.cuh>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_fill_empty_rows.cuh"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

struct BoolToSize {
  typedef int index_type;
  __device__ index_type operator()(bool x) const { return x ? 1 : 0; }
};

// Todo: According to the sorted_order, assign new value to output ptr.Calculate RevsereIndexMap.
// Input: values_ptr, indice_ptr, sorted_order, dense_row, default_value, emptyrow_count,
// input_row_end, output_values, output_indices.
// Output: output_values, output_indices
template <typename S>
__global__ void AssignValueKernel(Complex<S> *values_ptr, int64_t *indice_ptr, int64_t *sorted_order, size_t dense_row,
                                  Complex<S> *default_value, int *emptyrow_count, int64_t *input_row_end,
                                  Complex<S> *output_values, int64_t *output_indices, int indice_num,
                                  size_t *real_indice_num, int64_t *reverse_index_map) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dense_row; i += blockDim.x * gridDim.x) {
    if (i > 0 && input_row_end[i] == input_row_end[i - 1]) {
      // Empty row case. Calculate element in ith row by input_row_end[i]-input_row_end[i-1].
      int index = input_row_end[i] + emptyrow_count[i] - 1;
      output_values[index] = *default_value;
      output_indices[2 * index] = i;
      output_indices[2 * index + 1] = 0;
    } else if (i > 0 && input_row_end[i] > input_row_end[i - 1]) {
      // Not an empty row, calculate elements num and assign value to output_indice & output_value.
      for (int j = input_row_end[i - 1]; j < input_row_end[i]; j++) {
        int index_out = j + emptyrow_count[i];
        int index_in = sorted_order[j];
        output_values[index_out] = values_ptr[index_in];
        output_indices[2 * index_out] = indice_ptr[2 * index_in];
        output_indices[2 * index_out + 1] = indice_ptr[2 * index_in + 1];
        reverse_index_map[index_in] = index_out;
      }
    } else if (i == 0 && input_row_end[0] == 0) {
      // If the first row has no element.
      output_values[0] = *default_value;
      output_indices[0] = 0;
      output_indices[1] = 0;
      *real_indice_num = indice_num + emptyrow_count[dense_row - 1];
    } else if (i == 0 && input_row_end[0] > 0) {
      // The first row is not empty case.
      for (int j = 0; j < input_row_end[i]; j++) {
        int index_in = sorted_order[j];
        output_values[j] = values_ptr[index_in];
        output_indices[2 * j] = indice_ptr[2 * index_in];
        output_indices[2 * j + 1] = indice_ptr[2 * index_in + 1];
        reverse_index_map[index_in] = j;
      }
      *real_indice_num = indice_num + emptyrow_count[dense_row - 1];
    }
  }
  return;
}

// Todo: Calculate the elements num of each row.
// Input: dense_shape_ptr, row_indice, indice_num, cuda_stream
// Output: elements_per_row
__global__ void CalElementPerRowsKernel(int64_t *dense_shape_ptr, int64_t *indices_ptr, int64_t *elements_per_row,
                                        int indice_num) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indice_num; i += blockDim.x * gridDim.x) {
    int row = indices_ptr[i];
    MsAtomicAdd(&elements_per_row[row], static_cast<int64_t>(1));
  }
}

// Todo: Calculate output_empty_row_indicator_ptr.
// Input: elements_per_row, dense_row, output_empty_row_indicator_ptr.
// Output: output_empty_row_indicator_ptr.
__global__ void CalEmptyRowIndicatorKernel(int64_t *elements_per_row, size_t dense_row,
                                           bool *output_empty_row_indicator_ptr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dense_row; i += blockDim.x * gridDim.x) {
    if (elements_per_row[i] == 0) {
      output_empty_row_indicator_ptr[i] = 1;
    } else {
      output_empty_row_indicator_ptr[i] = 0;
    }
  }
}

// Todo: Extract row index in indice_ptr & Generate an ascend value index.
// Input: indices_ptr, row_indices, originorder, indice_num.
// Output: row_indices, originorder.
__global__ void CopyRowIndiceKernel(int64_t *indices_ptr, int64_t *row_indices, int64_t *origin_index, int indice_num) {
  int rank = 2;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < indice_num; i += blockDim.x * gridDim.x) {
    row_indices[i] = indices_ptr[i * rank];
    origin_index[i] = i;
  }
}

// Todo: Calculate the inclusive sum of empty_row_indicator.
// Input: dense_row, output_empty_row_indicator_ptr, empty_row_count_sum, cuda_stream.
// Output: empty_row_count_sum.
void InclusiveBoolPrefixSum(size_t dense_row, bool *output_empty_row_indicator_ptr, int *empty_row_count_sum,
                            cudaStream_t cuda_stream) {
  BoolToSize op;
  cub::TransformInputIterator<int, BoolToSize, const bool *> iter(output_empty_row_indicator_ptr, op);
  size_t temp_storage_bytes = 0;
  (void)cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, iter, empty_row_count_sum,
                                      static_cast<int>(dense_row), cuda_stream);
  void *d_temp_storage = nullptr;
  cudaStreamSynchronize(cuda_stream);
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, iter, empty_row_count_sum,
                                      static_cast<int>(dense_row), cuda_stream);
  cudaStreamSynchronize(cuda_stream);
  (void)cudaFree(d_temp_storage);
}

// Todo: Calculate the inclusive sum of elements_per_row.
// Input: dense_row, elements_per_row, input_row_ends, cuda_stream.
// Output: input_row_ends.
void InclusivePrefixSum(size_t dense_row, int64_t *elements_per_row, int64_t *input_row_ends,
                        cudaStream_t cuda_stream) {
  if (dense_row == 0) {
    return;
  }
  size_t temp_storage_bytes = 0;
  (void)cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, elements_per_row, input_row_ends,
                                      static_cast<int>(dense_row), cuda_stream);
  void *d_temp_storage = nullptr;
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, elements_per_row, input_row_ends,
                                      static_cast<int>(dense_row), cuda_stream);
  cudaStreamSynchronize(cuda_stream);
  (void)cudaFree(d_temp_storage);
}

// Todo: Sort the row_indice by key into ascend order, so we can get an key-value pair(origin index - sorted_order).
// Input: indice_size, cuda_stream, dense_shape_ptr, row_indices, origin_index, sorted_key, sorted_order, device_id.
// Output: sorted_key, sorted_order.
__host__ void RowsSort(int64_t indice_size, cudaStream_t cuda_stream, int64_t *dense_shape_ptr, int64_t *row_indices,
                       int64_t *origin_index, int64_t *sorted_key, int64_t *sorted_order, int device_id) {
  size_t temp_storage_ = 0;
  (void)cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_, row_indices, static_cast<int64_t *>(nullptr),
                                        origin_index, sorted_order, indice_size, 0, sizeof(int64_t) * 8, cuda_stream);
  void *d_temp_ = nullptr;
  (void)cudaMalloc(&d_temp_, temp_storage_);
  (void)cub::DeviceRadixSort::SortPairs(d_temp_, temp_storage_, row_indices, static_cast<int64_t *>(sorted_key),
                                        origin_index, sorted_order, indice_size, 0, sizeof(int64_t) * 8, cuda_stream);
  (void)cudaFree(d_temp_);
}

template <typename S>
__global__ void AssignValueKernel(S *values_ptr, int64_t *indice_ptr, int64_t *sorted_indices, size_t dense_row,
                                  S *default_value, int *emptyrow_count, int64_t *input_row_end, S *output_values,
                                  int64_t *output_indices, int indice_num, size_t *real_indice_num,
                                  int64_t *reverse_index_map) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dense_row; i += blockDim.x * gridDim.x) {
    if (i > 0 && input_row_end[i] == input_row_end[i - 1]) {
      // Empty row case. Calculate element in ith row by input_row_end[i]-input_row_end[i-1].
      int index = input_row_end[i] + emptyrow_count[i] - 1;
      output_values[index] = *default_value;
      output_indices[2 * index] = i;
      output_indices[2 * index + 1] = 0;
    } else if (i > 0 && input_row_end[i] > input_row_end[i - 1]) {
      // Not an empty row, calculate elements num and assign value to output_indice & output_value.
      for (int j = input_row_end[i - 1]; j < input_row_end[i]; j++) {
        int index_out = j + emptyrow_count[i];
        int index_in = sorted_indices[j];
        output_values[index_out] = values_ptr[index_in];
        output_indices[2 * index_out] = indice_ptr[2 * index_in];
        output_indices[2 * index_out + 1] = indice_ptr[2 * index_in + 1];
        reverse_index_map[index_in] = index_out;
      }
    } else if (i == 0 && input_row_end[0] == 0) {
      // If the first row has no element.
      output_values[0] = *default_value;
      output_indices[0] = 0;
      output_indices[1] = 0;
      *real_indice_num = indice_num + emptyrow_count[dense_row - 1];
    } else if (i == 0 && input_row_end[0] > 0) {
      // The first row is not empty case.
      for (int j = 0; j < input_row_end[i]; j++) {
        output_values[j] = values_ptr[sorted_indices[j]];
        output_indices[2 * j] = indice_ptr[2 * sorted_indices[j]];
        output_indices[2 * j + 1] = indice_ptr[2 * sorted_indices[j] + 1];
        reverse_index_map[sorted_indices[j]] = j;
      }
      *real_indice_num = indice_num + emptyrow_count[dense_row - 1];
    }
  }
  return;
}
template <typename S>
CUDA_LIB_EXPORT cudaError_t SparseFillEmptyRows(int64_t *indices_ptr, Complex<S> *values_ptr, Complex<S> *default_value,
                                                int64_t *dense_shape_ptr, int device_id, int indice_num,
                                                size_t dense_row, int64_t *elements_per_rows, int *empty_row_count_sum,
                                                int64_t *row_indices, int64_t *input_row_ends, int64_t *sorted_indices,
                                                size_t *final_shape, int64_t *origin_index, int64_t *sorted_key,
                                                cudaStream_t cuda_stream, int64_t *output_indices_ptr,
                                                Complex<S> *output_values_ptr, bool *output_empty_row_indicator_ptr,
                                                int64_t *output_reverse_index_map_ptr) {
  int thread_num_dense_row = 256 < dense_row ? 256 : dense_row;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((dense_row - 1) / thread_num_dense_row) + 1), max_blocks);
  CopyRowIndiceKernel<<<block_num, thread_num_dense_row, 0, cuda_stream>>>(indices_ptr, row_indices, origin_index,
                                                                           indice_num);
  cudaMemset(elements_per_rows, 0, dense_row * sizeof(int64_t));
  int thread_num_indice_num = 256 < indice_num ? 256 : indice_num;
  block_num = std::min(static_cast<int>(((indice_num - 1) / thread_num_indice_num) + 1), max_blocks);
  CalElementPerRowsKernel<<<block_num, thread_num_indice_num, 0, cuda_stream>>>(dense_shape_ptr, row_indices,
                                                                                elements_per_rows, indice_num);
  CalEmptyRowIndicatorKernel<<<block_num, thread_num_dense_row, 0, cuda_stream>>>(elements_per_rows, dense_row,
                                                                                  output_empty_row_indicator_ptr);
  InclusivePrefixSum(dense_row, elements_per_rows, input_row_ends, cuda_stream);
  InclusiveBoolPrefixSum(dense_row, output_empty_row_indicator_ptr, empty_row_count_sum, cuda_stream);
  RowsSort(indice_num, cuda_stream, dense_shape_ptr, row_indices, origin_index, sorted_key, sorted_indices, device_id);
  AssignValueKernel<<<block_num, thread_num_dense_row, 0, cuda_stream>>>(
    values_ptr, indices_ptr, sorted_indices, dense_row, default_value, empty_row_count_sum, input_row_ends,
    output_values_ptr, output_indices_ptr, indice_num, final_shape, output_reverse_index_map_ptr);
  return GetCudaStatus();
}

template <typename S>
CUDA_LIB_EXPORT cudaError_t SparseFillEmptyRows(int64_t *indices_ptr, S *values_ptr, S *default_value,
                                                int64_t *dense_shape_ptr, int device_id, int indice_num,
                                                size_t dense_row, int64_t *elements_per_rows, int *empty_row_count_sum,
                                                int64_t *row_indices, int64_t *input_row_ends, int64_t *sorted_indices,
                                                size_t *final_shape, int64_t *origin_index, int64_t *sorted_key,
                                                cudaStream_t cuda_stream, int64_t *output_indices_ptr,
                                                S *output_values_ptr, bool *output_empty_row_indicator_ptr,
                                                int64_t *output_reverse_index_map_ptr) {
  int thread_num_dense_row = 256 < dense_row ? 256 : dense_row;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((dense_row - 1) / thread_num_dense_row) + 1), max_blocks);
  CopyRowIndiceKernel<<<block_num, thread_num_dense_row, 0, cuda_stream>>>(indices_ptr, row_indices, origin_index,
                                                                           indice_num);
  cudaMemset(elements_per_rows, 0, dense_row * sizeof(int64_t));
  int thread_num_indice_num = 256 < indice_num ? 256 : indice_num;
  block_num = std::min(static_cast<int>(((indice_num - 1) / thread_num_indice_num) + 1), max_blocks);
  CalElementPerRowsKernel<<<block_num, thread_num_indice_num, 0, cuda_stream>>>(dense_shape_ptr, row_indices,
                                                                                elements_per_rows, indice_num);
  CalEmptyRowIndicatorKernel<<<block_num, thread_num_dense_row, 0, cuda_stream>>>(elements_per_rows, dense_row,
                                                                                  output_empty_row_indicator_ptr);
  InclusivePrefixSum(dense_row, elements_per_rows, input_row_ends, cuda_stream);
  InclusiveBoolPrefixSum(dense_row, output_empty_row_indicator_ptr, empty_row_count_sum, cuda_stream);
  RowsSort(indice_num, cuda_stream, dense_shape_ptr, row_indices, origin_index, sorted_key, sorted_indices, device_id);
  AssignValueKernel<<<block_num, thread_num_dense_row, 0, cuda_stream>>>(
    values_ptr, indices_ptr, sorted_indices, dense_row, default_value, empty_row_count_sum, input_row_ends,
    output_values_ptr, output_indices_ptr, indice_num, final_shape, output_reverse_index_map_ptr);
  return GetCudaStatus();
}

#define TEMPLATE_INSTANCE(DTYPE)                                                                                       \
  template CUDA_LIB_EXPORT cudaError_t SparseFillEmptyRows<DTYPE>(                                                     \
    int64_t * indices_ptr, DTYPE * values_ptr, DTYPE * default_value, int64_t * dense_shape_ptr, int device_id,        \
    int indice_num, size_t dense_row, int64_t *elements_per_rows, int *rows_are_not_ordered, int64_t *row_indices,     \
    int64_t *input_row_ends, int64_t *sorted_indices, size_t *final_shape, int64_t *origin_index, int64_t *sorted_key, \
    cudaStream_t cuda_stream, int64_t *output_indices_ptr, DTYPE *output_values_ptr,                                   \
    bool *output_empty_row_indicator_ptr, int64_t *output_reverse_index_map_ptr);

TEMPLATE_INSTANCE(float)
TEMPLATE_INSTANCE(half)
TEMPLATE_INSTANCE(double)
TEMPLATE_INSTANCE(int)
TEMPLATE_INSTANCE(int64_t)
TEMPLATE_INSTANCE(uint32_t)
TEMPLATE_INSTANCE(uint64_t)
TEMPLATE_INSTANCE(uint16_t)
TEMPLATE_INSTANCE(uint8_t)
TEMPLATE_INSTANCE(int8_t)
TEMPLATE_INSTANCE(int16_t)
TEMPLATE_INSTANCE(bool)
TEMPLATE_INSTANCE(Complex<float>)
TEMPLATE_INSTANCE(Complex<double>)
