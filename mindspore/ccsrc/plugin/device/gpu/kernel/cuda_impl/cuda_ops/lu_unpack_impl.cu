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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lu_unpack_impl.cuh"

template <typename S>
__global__ void InitOrder(const size_t p_size, const int64_t lu_data_dim1, S *final_order) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < p_size * lu_data_dim1; pos += gridDim.x * blockDim.x) {
    final_order[pos] = pos % lu_data_dim1;
  }
}

template <typename S>
__global__ void SwapOrder(const size_t p_size, const int64_t lu_data_dim1, const int64_t lu_pivots_dim,
                          S *lu_pivots_ptr, S *final_order) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < p_size; pos += gridDim.x * blockDim.x) {
    S *lu_pivots_working_ptr = lu_pivots_ptr + pos * lu_pivots_dim;
    for (S idx = 0; idx < lu_pivots_dim; ++idx) {
      S perm_pivots_id = lu_pivots_working_ptr[idx] - 1;
      S tmp = final_order[pos * lu_data_dim1 + idx];
      final_order[pos * lu_data_dim1 + idx] = final_order[pos * lu_data_dim1 + perm_pivots_id];
      final_order[pos * lu_data_dim1 + perm_pivots_id] = tmp;
    }
  }
}

template <typename T, typename S>
__global__ void AssignEyeValue(const size_t p_size, const size_t lu_data_dim1, S *final_order, T *pivots_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < p_size * lu_data_dim1; pos += gridDim.x * blockDim.x) {
    size_t row_start = (pos / lu_data_dim1 * lu_data_dim1 + final_order[pos]) * lu_data_dim1;
    size_t eye_idx = pos % lu_data_dim1;
    *(pivots_ptr + row_start + eye_idx) = static_cast<T>(1);
  }
  return;
}

template <typename T, typename S>
cudaError_t TransposeEyeMatrix(S *lu_pivots_ptr, T *pivots_ptr, S *final_order, const int64_t batch_num,
                               const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id,
                               cudaStream_t cuda_stream) {
  cudaMemset(pivots_ptr, 0, batch_num * lu_data_dim1 * lu_data_dim1 * sizeof(T));
  InitOrder<<<CUDA_BLOCKS(device_id, batch_num * lu_data_dim1), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    batch_num, lu_data_dim1, final_order);
  SwapOrder<<<CUDA_BLOCKS(device_id, batch_num), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    batch_num, lu_data_dim1, lu_pivots_dim, lu_pivots_ptr, final_order);
  AssignEyeValue<<<CUDA_BLOCKS(device_id, batch_num * lu_data_dim1), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    batch_num, lu_data_dim1, final_order, pivots_ptr);
  return GetCudaStatus();
}

template <typename T>
__global__ void TriuAux(const size_t size, const T *input, const int64_t out_row, const int64_t out_col,
                        const int64_t lu_row, const int64_t lu_col, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t out_matrix_size = out_row * out_col;
    int64_t in_matrix_size = lu_row * lu_col;
    int64_t idx_row = pos % out_matrix_size / out_col;
    int64_t idx_col = pos % out_matrix_size % out_col;

    if (idx_row <= idx_col) {
      int64_t batch_size = pos / out_matrix_size;
      int64_t in_pos = batch_size * in_matrix_size + idx_row * lu_col + idx_col;
      output[pos] = input[in_pos];
    } else {
      output[pos] = static_cast<T>(0.0);
    }
  }
  return;
}

template <typename T>
cudaError_t CalTriuAux(const size_t size, const T *input, const int64_t out_row, const int64_t out_col,
                       const int64_t lu_row, const int64_t lu_col, T *output, const uint32_t &device_id,
                       cudaStream_t cuda_stream) {
  TriuAux<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, out_row, out_col,
                                                                                     lu_row, lu_col, output);
  return GetCudaStatus();
}

template <typename T>
__global__ void TrilAux(const size_t size, const T *input, const int64_t out_row, const int64_t out_col,
                        const int64_t lu_row, const int64_t lu_col, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t out_matrix_size = out_row * out_col;
    int64_t in_matrix_size = lu_row * lu_col;
    int64_t idx_row = pos % out_matrix_size / out_col;
    int64_t idx_col = pos % out_matrix_size % out_col;

    if (idx_row == idx_col) {
      output[pos] = static_cast<T>(1.0);
    } else if (idx_row > idx_col) {
      int64_t batch_size = pos / out_matrix_size;
      int64_t in_pos = batch_size * in_matrix_size + idx_row * lu_col + idx_col;
      output[pos] = input[in_pos];
    } else {
      output[pos] = static_cast<T>(0.0);
    }
  }
  return;
}

template <typename T>
cudaError_t CalTrilAux(const size_t size, const T *input, const int64_t out_row, const int64_t out_col,
                       const int64_t lu_row, const int64_t lu_col, T *output, const uint32_t &device_id,
                       cudaStream_t cuda_stream) {
  TrilAux<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, out_row, out_col,
                                                                                     lu_row, lu_col, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<double, int64_t>(
  int64_t *lu_pivots_ptr, double *pivots_ptr, int64_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<double, int32_t>(
  int32_t *lu_pivots_ptr, double *pivots_ptr, int32_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<double, int16_t>(
  int16_t *lu_pivots_ptr, double *pivots_ptr, int16_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<double, int8_t>(
  int8_t *lu_pivots_ptr, double *pivots_ptr, int8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<double, uint8_t>(
  uint8_t *lu_pivots_ptr, double *pivots_ptr, uint8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<float, int64_t>(
  int64_t *lu_pivots_ptr, float *pivots_ptr, int64_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<float, int32_t>(
  int32_t *lu_pivots_ptr, float *pivots_ptr, int32_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<float, int16_t>(
  int16_t *lu_pivots_ptr, float *pivots_ptr, int16_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<float, int8_t>(
  int8_t *lu_pivots_ptr, float *pivots_ptr, int8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<float, uint8_t>(
  uint8_t *lu_pivots_ptr, float *pivots_ptr, uint8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<half, int64_t>(
  int64_t *lu_pivots_ptr, half *pivots_ptr, int64_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<half, int32_t>(
  int32_t *lu_pivots_ptr, half *pivots_ptr, int32_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<half, int16_t>(
  int16_t *lu_pivots_ptr, half *pivots_ptr, int16_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<half, int8_t>(
  int8_t *lu_pivots_ptr, half *pivots_ptr, int8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<half, uint8_t>(
  uint8_t *lu_pivots_ptr, half *pivots_ptr, uint8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int64_t, int64_t>(
  int64_t *lu_pivots_ptr, int64_t *pivots_ptr, int64_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int64_t, int32_t>(
  int32_t *lu_pivots_ptr, int64_t *pivots_ptr, int32_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int64_t, int16_t>(
  int16_t *lu_pivots_ptr, int64_t *pivots_ptr, int16_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int64_t, int8_t>(
  int8_t *lu_pivots_ptr, int64_t *pivots_ptr, int8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int64_t, uint8_t>(
  uint8_t *lu_pivots_ptr, int64_t *pivots_ptr, uint8_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int32_t, int64_t>(
  int64_t *lu_pivots_ptr, int32_t *pivots_ptr, int64_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int32_t, int32_t>(
  int32_t *lu_pivots_ptr, int32_t *pivots_ptr, int32_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int32_t, int16_t>(
  int16_t *lu_pivots_ptr, int32_t *pivots_ptr, int16_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int32_t, int8_t>(
  int8_t *lu_pivots_ptr, int32_t *pivots_ptr, int8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int32_t, uint8_t>(
  uint8_t *lu_pivots_ptr, int32_t *pivots_ptr, uint8_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int16_t, int64_t>(
  int64_t *lu_pivots_ptr, int16_t *pivots_ptr, int64_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int16_t, int32_t>(
  int32_t *lu_pivots_ptr, int16_t *pivots_ptr, int32_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int16_t, int16_t>(
  int16_t *lu_pivots_ptr, int16_t *pivots_ptr, int16_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int16_t, int8_t>(
  int8_t *lu_pivots_ptr, int16_t *pivots_ptr, int8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int16_t, uint8_t>(
  uint8_t *lu_pivots_ptr, int16_t *pivots_ptr, uint8_t *final_order, const int64_t batch_num,
  const int64_t lu_data_dim1, const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int8_t, int64_t>(
  int64_t *lu_pivots_ptr, int8_t *pivots_ptr, int64_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int8_t, int32_t>(
  int32_t *lu_pivots_ptr, int8_t *pivots_ptr, int32_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int8_t, int16_t>(
  int16_t *lu_pivots_ptr, int8_t *pivots_ptr, int16_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int8_t, int8_t>(
  int8_t *lu_pivots_ptr, int8_t *pivots_ptr, int8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix<int8_t, uint8_t>(
  uint8_t *lu_pivots_ptr, int8_t *pivots_ptr, uint8_t *final_order, const int64_t batch_num, const int64_t lu_data_dim1,
  const int64_t lu_pivots_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalTriuAux<uint8_t>(const size_t size, const uint8_t *input, const int64_t out_row,
                                                         const int64_t out_col, const int64_t lu_row,
                                                         const int64_t lu_col, uint8_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<uint16_t>(const size_t size, const uint16_t *input,
                                                          const int64_t out_row, const int64_t out_col,
                                                          const int64_t lu_row, const int64_t lu_col, uint16_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<uint32_t>(const size_t size, const uint32_t *input,
                                                          const int64_t out_row, const int64_t out_col,
                                                          const int64_t lu_row, const int64_t lu_col, uint32_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<uint64_t>(const size_t size, const uint64_t *input,
                                                          const int64_t out_row, const int64_t out_col,
                                                          const int64_t lu_row, const int64_t lu_col, uint64_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<int8_t>(const size_t size, const int8_t *input, const int64_t out_row,
                                                        const int64_t out_col, const int64_t lu_row,
                                                        const int64_t lu_col, int8_t *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<int16_t>(const size_t size, const int16_t *input, const int64_t out_row,
                                                         const int64_t out_col, const int64_t lu_row,
                                                         const int64_t lu_col, int16_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<int>(const size_t size, const int *input, const int64_t out_row,
                                                     const int64_t out_col, const int64_t lu_row, const int64_t lu_col,
                                                     int *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<int64_t>(const size_t size, const int64_t *input, const int64_t out_row,
                                                         const int64_t out_col, const int64_t lu_row,
                                                         const int64_t lu_col, int64_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<half>(const size_t size, const half *input, const int64_t out_row,
                                                      const int64_t out_col, const int64_t lu_row, const int64_t lu_col,
                                                      half *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<float>(const size_t size, const float *input, const int64_t out_row,
                                                       const int64_t out_col, const int64_t lu_row,
                                                       const int64_t lu_col, float *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<double>(const size_t size, const double *input, const int64_t out_row,
                                                        const int64_t out_col, const int64_t lu_row,
                                                        const int64_t lu_col, double *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuAux<bool>(const size_t size, const bool *input, const int64_t out_row,
                                                      const int64_t out_col, const int64_t lu_row, const int64_t lu_col,
                                                      bool *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalTrilAux<uint8_t>(const size_t size, const uint8_t *input, const int64_t out_row,
                                                         const int64_t out_col, const int64_t lu_row,
                                                         const int64_t lu_col, uint8_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<uint16_t>(const size_t size, const uint16_t *input,
                                                          const int64_t out_row, const int64_t out_col,
                                                          const int64_t lu_row, const int64_t lu_col, uint16_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<uint32_t>(const size_t size, const uint32_t *input,
                                                          const int64_t out_row, const int64_t out_col,
                                                          const int64_t lu_row, const int64_t lu_col, uint32_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<uint64_t>(const size_t size, const uint64_t *input,
                                                          const int64_t out_row, const int64_t out_col,
                                                          const int64_t lu_row, const int64_t lu_col, uint64_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<int8_t>(const size_t size, const int8_t *input, const int64_t out_row,
                                                        const int64_t out_col, const int64_t lu_row,
                                                        const int64_t lu_col, int8_t *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<int16_t>(const size_t size, const int16_t *input, const int64_t out_row,
                                                         const int64_t out_col, const int64_t lu_row,
                                                         const int64_t lu_col, int16_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<int>(const size_t size, const int *input, const int64_t out_row,
                                                     const int64_t out_col, const int64_t lu_row, const int64_t lu_col,
                                                     int *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<int64_t>(const size_t size, const int64_t *input, const int64_t out_row,
                                                         const int64_t out_col, const int64_t lu_row,
                                                         const int64_t lu_col, int64_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<half>(const size_t size, const half *input, const int64_t out_row,
                                                      const int64_t out_col, const int64_t lu_row, const int64_t lu_col,
                                                      half *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<float>(const size_t size, const float *input, const int64_t out_row,
                                                       const int64_t out_col, const int64_t lu_row,
                                                       const int64_t lu_col, float *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<double>(const size_t size, const double *input, const int64_t out_row,
                                                        const int64_t out_col, const int64_t lu_row,
                                                        const int64_t lu_col, double *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilAux<bool>(const size_t size, const bool *input, const int64_t out_row,
                                                      const int64_t out_col, const int64_t lu_row, const int64_t lu_col,
                                                      bool *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
