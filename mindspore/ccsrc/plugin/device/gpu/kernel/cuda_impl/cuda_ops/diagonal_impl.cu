/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <complex.h>
#include "diagonal_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename R>
using Complex = mindspore::utils::Complex<R>;

const size_t DIM_SIZE = 20;
struct DiagonalInfo {
  int64_t input_strides[DIM_SIZE];
  int64_t output_strides[DIM_SIZE];
  int64_t y_strides[DIM_SIZE];
  int64_t output_shape[DIM_SIZE];
};

template <typename T>
__global__ void DiagonalKernel(const int64_t size, const T *input, const int64_t shape_size,
                               const int64_t storage_offset, DiagonalInfo diagonalInfo, T *output) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t input_pos = 0;
    int64_t temp = pos;
    int64_t cur_index = 0;
    for (int64_t i = shape_size - 1; i >= 0; i--) {
      cur_index = temp / diagonalInfo.y_strides[i] % diagonalInfo.output_shape[i];
      temp -= cur_index * diagonalInfo.y_strides[i];
      input_pos += cur_index * diagonalInfo.output_strides[i];
    }
    input_pos += storage_offset;
    output[pos] = input[input_pos];
  }
}

template <typename T>
cudaError_t CalDiagonal(const T *input, const int64_t offset_, const int64_t dim1_, const int64_t dim2_,
                        std::vector<int64_t> input_shape, const std::vector<int64_t> output_shape, T *output,
                        const uint32_t &device_id, cudaStream_t cuda_stream) {
  int64_t input_size = input_shape.size();
  int64_t shape_size = output_shape.size();
  int64_t index = 0;
  int64_t size = 1;
  int64_t storage_offset = 0;
  DiagonalInfo diagonalInfo;
  diagonalInfo.input_strides[input_size - 1] = 1;
  diagonalInfo.y_strides[shape_size - 1] = 1;

  for (int64_t i = 0; i < shape_size; i++) {
    size *= output_shape[i];
    diagonalInfo.output_shape[i] = output_shape[i];
  }

  for (int64_t i = input_size - 2; i >= 0; i--) {
    diagonalInfo.input_strides[i] = diagonalInfo.input_strides[i + 1] * input_shape[i + 1];
  }

  for (int64_t i = shape_size - 2; i >= 0; i--) {
    diagonalInfo.y_strides[i] = diagonalInfo.y_strides[i + 1] * output_shape[i + 1];
  }

  for (int64_t tmp_dim = 0; tmp_dim < input_size; tmp_dim++) {
    if (tmp_dim != dim1_ && tmp_dim != dim2_) {
      diagonalInfo.output_strides[index] = diagonalInfo.input_strides[tmp_dim];
      index += 1;
    }
  }
  diagonalInfo.output_strides[shape_size - 1] = diagonalInfo.input_strides[dim1_] + diagonalInfo.input_strides[dim2_];

  if (offset_ >= 0) {
    storage_offset += offset_ * diagonalInfo.input_strides[dim2_];
  } else {
    storage_offset -= offset_ * diagonalInfo.input_strides[dim1_];
  }

  DiagonalKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, shape_size, storage_offset, diagonalInfo, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalDiagonal<double>(const double *input, const int64_t offset_,
                                                         const int64_t dim1_, const int64_t dim2_,
                                                         const std::vector<int64_t> input_shape,
                                                         const std::vector<int64_t> output_shape, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<float>(const float *input, const int64_t offset_, const int64_t dim1_,
                                                        const int64_t dim2_, const std::vector<int64_t> input_shape,
                                                        const std::vector<int64_t> output_shape, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<bool>(const bool *input, const int64_t offset_, const int64_t dim1_,
                                                        const int64_t dim2_, const std::vector<int64_t> input_shape,
                                                        const std::vector<int64_t> output_shape, bool *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<half>(const half *input, const int64_t offset_, const int64_t dim1_,
                                                       const int64_t dim2_, const std::vector<int64_t> input_shape,
                                                       const std::vector<int64_t> output_shape, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<int8_t>(const int8_t *input, const int64_t offset_,
                                                         const int64_t dim1_, const int64_t dim2_,
                                                         const std::vector<int64_t> input_shape,
                                                         const std::vector<int64_t> output_shape, int8_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<int16_t>(const int16_t *input, const int64_t offset_,
                                                          const int64_t dim1_, const int64_t dim2_,
                                                          const std::vector<int64_t> input_shape,
                                                          const std::vector<int64_t> output_shape, int16_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<int32_t>(const int32_t *input, const int64_t offset_,
                                                          const int64_t dim1_, const int64_t dim2_,
                                                          const std::vector<int64_t> input_shape,
                                                          const std::vector<int64_t> output_shape, int32_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<int64_t>(const int64_t *input, const int64_t offset_,
                                                          const int64_t dim1_, const int64_t dim2_,
                                                          const std::vector<int64_t> input_shape,
                                                          const std::vector<int64_t> output_shape, int64_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<uint8_t>(const uint8_t *input, const int64_t offset_,
                                                          const int64_t dim1_, const int64_t dim2_,
                                                          const std::vector<int64_t> input_shape,
                                                          const std::vector<int64_t> output_shape, uint8_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<uint16_t>(const uint16_t *input, const int64_t offset_,
                                                           const int64_t dim1_, const int64_t dim2_,
                                                           const std::vector<int64_t> input_shape,
                                                           const std::vector<int64_t> output_shape, uint16_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<uint32_t>(const uint32_t *input, const int64_t offset_,
                                                           const int64_t dim1_, const int64_t dim2_,
                                                           const std::vector<int64_t> input_shape,
                                                           const std::vector<int64_t> output_shape, uint32_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<uint64_t>(const uint64_t *input, const int64_t offset_,
                                                           const int64_t dim1_, const int64_t dim2_,
                                                           const std::vector<int64_t> input_shape,
                                                           const std::vector<int64_t> output_shape, uint64_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
