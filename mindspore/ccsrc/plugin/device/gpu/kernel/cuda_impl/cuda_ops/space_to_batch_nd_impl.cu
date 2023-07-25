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
#include <cuda_runtime.h>
#include "space_to_batch_nd_impl.cuh"

__constant__ int64_t con_paddings_start[8];
__constant__ int64_t con_block_shape[8];
__constant__ int64_t con_input_shape[8];
__constant__ int64_t con_stride[8];
__constant__ int64_t con_on_stride[8];

template <typename T>
__global__ void SpaceToBatchND(const T *__restrict__ input, const int64_t *paddings_start, const int64_t *block_shape,
                               const int64_t *input_shape, const size_t input_shape_size, const size_t off_set,
                               const size_t input_size_, T *__restrict__ output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size_; pos += gridDim.x * blockDim.x) {
    int64_t cur_pos = pos;
    int idx = input_shape_size - 1;
    int64_t out_pos = 0;
    int64_t idx_on = 0;
    int64_t cur_out_idx = 0;
    int64_t cur_inp_idx = 0;
    int64_t temp_idx = 0;
    int64_t offset_idx = 0;
    for (; idx >= off_set; --idx) {
      cur_out_idx = cur_pos % con_input_shape[idx];
      cur_pos /= con_input_shape[idx];
      offset_idx = idx - off_set;
      temp_idx = (cur_out_idx + con_paddings_start[offset_idx]);
      cur_inp_idx = temp_idx / con_block_shape[offset_idx];
      out_pos += cur_inp_idx * con_stride[idx];
      idx_on += (temp_idx % con_block_shape[offset_idx]) * con_on_stride[offset_idx];
    }
    for (; idx > 0; --idx) {
      cur_out_idx = cur_pos % con_input_shape[idx];
      cur_pos /= con_input_shape[idx];
      out_pos += cur_out_idx * con_stride[idx];
    }

    cur_inp_idx = idx_on * con_input_shape[0] + cur_pos % con_input_shape[0];

    out_pos += cur_inp_idx * con_stride[0];
    output[out_pos] = input[pos];
  }
  return;
}

template <typename T>
cudaError_t CalSpaceToBatchND(const T *input, const int64_t *paddings_start, const int64_t *block_shape,
                              const int64_t *input_shape, const size_t input_shape_size, const int64_t *stride,
                              const int64_t *on_stride, const size_t off_set, const size_t input_size_,
                              const size_t output_size_, T *output, const uint32_t &device_id,
                              cudaStream_t cuda_stream) {
  cudaMemset(output, 0, output_size_ * sizeof(T));
  cudaMemcpyToSymbol(con_input_shape, input_shape, sizeof(int64_t) * input_shape_size);
  cudaMemcpyToSymbol(con_block_shape, block_shape, sizeof(int64_t) * (input_shape_size - off_set));
  cudaMemcpyToSymbol(con_paddings_start, paddings_start, sizeof(int64_t) * (input_shape_size - off_set));
  cudaMemcpyToSymbol(con_on_stride, on_stride, sizeof(int64_t) * (input_shape_size - off_set));
  cudaMemcpyToSymbol(con_stride, stride, sizeof(int64_t) * input_shape_size);
  SpaceToBatchND<<<CUDA_BLOCKS(device_id, input_size_), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, paddings_start, block_shape, input_shape, input_shape_size, off_set, input_size_, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<uint8_t>(const uint8_t *input, const int64_t *paddings_start,
                                                                const int64_t *block_shape, const int64_t *input_shape,
                                                                const size_t input_shape_size, const int64_t *stride,
                                                                const int64_t *on_stride, const size_t off_set,
                                                                const size_t input_size_, const size_t output_size_,
                                                                uint8_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<uint16_t>(const uint16_t *input, const int64_t *paddings_start,
                                                                 const int64_t *block_shape, const int64_t *input_shape,
                                                                 const size_t input_shape_size, const int64_t *stride,
                                                                 const int64_t *on_stride, const size_t off_set,
                                                                 const size_t input_size_, const size_t output_size_,
                                                                 uint16_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<uint32_t>(const uint32_t *input, const int64_t *paddings_start,
                                                                 const int64_t *block_shape, const int64_t *input_shape,
                                                                 const size_t input_shape_size, const int64_t *stride,
                                                                 const int64_t *on_stride, const size_t off_set,
                                                                 const size_t input_size_, const size_t output_size_,
                                                                 uint32_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<uint64_t>(const uint64_t *input, const int64_t *paddings_start,
                                                                 const int64_t *block_shape, const int64_t *input_shape,
                                                                 const size_t input_shape_size, const int64_t *stride,
                                                                 const int64_t *on_stride, const size_t off_set,
                                                                 const size_t input_size_, const size_t output_size_,
                                                                 uint64_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<int8_t>(const int8_t *input, const int64_t *paddings_start,
                                                               const int64_t *block_shape, const int64_t *input_shape,
                                                               const size_t input_shape_size, const int64_t *stride,
                                                               const int64_t *on_stride, const size_t off_set,
                                                               const size_t input_size_, const size_t output_size_,
                                                               int8_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<int16_t>(const int16_t *input, const int64_t *paddings_start,
                                                                const int64_t *block_shape, const int64_t *input_shape,
                                                                const size_t input_shape_size, const int64_t *stride,
                                                                const int64_t *on_stride, const size_t off_set,
                                                                const size_t input_size_, const size_t output_size_,
                                                                int16_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<int32_t>(const int32_t *input, const int64_t *paddings_start,
                                                                const int64_t *block_shape, const int64_t *input_shape,
                                                                const size_t input_shape_size, const int64_t *stride,
                                                                const int64_t *on_stride, const size_t off_set,
                                                                const size_t input_size_, const size_t output_size_,
                                                                int32_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<int64_t>(const int64_t *input, const int64_t *paddings_start,
                                                                const int64_t *block_shape, const int64_t *input_shape,
                                                                const size_t input_shape_size, const int64_t *stride,
                                                                const int64_t *on_stride, const size_t off_set,
                                                                const size_t input_size_, const size_t output_size_,
                                                                int64_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<half>(const half *input, const int64_t *paddings_start,
                                                             const int64_t *block_shape, const int64_t *input_shape,
                                                             const size_t input_shape_size, const int64_t *stride,
                                                             const int64_t *on_stride, const size_t off_set,
                                                             const size_t input_size_, const size_t output_size_,
                                                             half *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<float>(const float *input, const int64_t *paddings_start,
                                                              const int64_t *block_shape, const int64_t *input_shape,
                                                              const size_t input_shape_size, const int64_t *stride,
                                                              const int64_t *on_stride, const size_t off_set,
                                                              const size_t input_size_, const size_t output_size_,
                                                              float *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatchND<double>(const double *input, const int64_t *paddings_start,
                                                               const int64_t *block_shape, const int64_t *input_shape,
                                                               const size_t input_shape_size, const int64_t *stride,
                                                               const int64_t *on_stride, const size_t off_set,
                                                               const size_t input_size_, const size_t output_size_,
                                                               double *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
