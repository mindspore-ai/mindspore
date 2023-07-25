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
#include "batch_to_space_nd_impl.cuh"

__constant__ int64_t con_crop_start[8];
__constant__ int64_t con_block_shape[8];
__constant__ int64_t con_output_shape[8];
__constant__ int64_t stride[8];
__constant__ int64_t on_stride[8];

template <typename T>
__global__ void BatchToSpaceND(const T *__restrict__ input, const int64_t *crops_start, const int64_t *block_shape,
                               const int64_t *output_shape, const size_t output_shape_size, const size_t off_set_,
                               const size_t output_size_, T *__restrict__ output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size_; pos += gridDim.x * blockDim.x) {
    int64_t cur_pos = pos;
    int idx = output_shape_size - 1;
    int64_t in_pos = 0;
    int64_t idx_on = 0;
    int64_t cur_out_idx = 0;
    int64_t cur_inp_idx = 0;
    int64_t temp_idx = 0;
    int64_t offset_idx = 0;
    for (; idx >= off_set_; --idx) {
      cur_out_idx = cur_pos % con_output_shape[idx];
      cur_pos /= con_output_shape[idx];
      offset_idx = idx - off_set_;
      temp_idx = (cur_out_idx + con_crop_start[offset_idx]);
      cur_inp_idx = temp_idx / con_block_shape[offset_idx];
      in_pos += cur_inp_idx * stride[idx];
      idx_on += (temp_idx % con_block_shape[offset_idx]) * on_stride[offset_idx];
    }
    for (; idx > 0; --idx) {
      cur_out_idx = cur_pos % con_output_shape[idx];
      cur_pos /= con_output_shape[idx];
      in_pos += cur_out_idx * stride[idx];
    }

    cur_inp_idx = idx_on * con_output_shape[0] + cur_pos % con_output_shape[0];

    in_pos += cur_inp_idx * stride[0];
    output[pos] = input[in_pos];
  }
  return;
}

template <typename T>
cudaError_t CalBatchToSpaceND(const T *input, const int64_t *crops_start, const int64_t *block_shape,
                              const int64_t *output_shape, const size_t output_shape_size, const int64_t *stride_,
                              const int64_t *on_stride_, const size_t off_set_, const size_t output_size_, T *output,
                              const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaMemcpyToSymbol(con_output_shape, output_shape, sizeof(int64_t) * output_shape_size);
  cudaMemcpyToSymbol(con_block_shape, block_shape, sizeof(int64_t) * (output_shape_size - off_set_));
  cudaMemcpyToSymbol(con_crop_start, crops_start, sizeof(int64_t) * (output_shape_size - off_set_));
  cudaMemcpyToSymbol(stride, stride_, sizeof(int64_t) * output_shape_size);
  cudaMemcpyToSymbol(on_stride, on_stride_, sizeof(int64_t) * (output_shape_size - off_set_));
  BatchToSpaceND<<<CUDA_BLOCKS(device_id, output_size_), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, crops_start, block_shape, output_shape, output_shape_size, off_set_, output_size_, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<int8_t>(const int8_t *input, const int64_t *crops_start,
                                                               const int64_t *block_shape, const int64_t *output_shape,
                                                               const size_t output_shape_size, const int64_t *stride_,
                                                               const int64_t *on_stride_, const size_t off_set_,
                                                               const size_t output_size_, int8_t *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<int16_t>(const int16_t *input, const int64_t *crops_start,
                                                                const int64_t *block_shape, const int64_t *output_shape,
                                                                const size_t output_shape_size, const int64_t *stride_,
                                                                const int64_t *on_stride_, const size_t off_set_,
                                                                const size_t output_size_, int16_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<int32_t>(const int32_t *input, const int64_t *crops_start,
                                                                const int64_t *block_shape, const int64_t *output_shape,
                                                                const size_t output_shape_size, const int64_t *stride_,
                                                                const int64_t *on_stride_, const size_t off_set_,
                                                                const size_t output_size_, int32_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<int64_t>(const int64_t *input, const int64_t *crops_start,
                                                                const int64_t *block_shape, const int64_t *output_shape,
                                                                const size_t output_shape_size, const int64_t *stride_,
                                                                const int64_t *on_stride_, const size_t off_set_,
                                                                const size_t output_size_, int64_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<uint8_t>(const uint8_t *input, const int64_t *crops_start,
                                                                const int64_t *block_shape, const int64_t *output_shape,
                                                                const size_t output_shape_size, const int64_t *stride_,
                                                                const int64_t *on_stride_, const size_t off_set_,
                                                                const size_t output_size_, uint8_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<uint16_t>(
  const uint16_t *input, const int64_t *crops_start, const int64_t *block_shape, const int64_t *output_shape,
  const size_t output_shape_size, const int64_t *stride_, const int64_t *on_stride_, const size_t off_set_,
  const size_t output_size_, uint16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<uint32_t>(
  const uint32_t *input, const int64_t *crops_start, const int64_t *block_shape, const int64_t *output_shape,
  const size_t output_shape_size, const int64_t *stride_, const int64_t *on_stride_, const size_t off_set_,
  const size_t output_size_, uint32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<uint64_t>(
  const uint64_t *input, const int64_t *crops_start, const int64_t *block_shape, const int64_t *output_shape,
  const size_t output_shape_size, const int64_t *stride_, const int64_t *on_stride_, const size_t off_set_,
  const size_t output_size_, uint64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<half>(const half *input, const int64_t *crops_start,
                                                             const int64_t *block_shape, const int64_t *output_shape,
                                                             const size_t output_shape_size, const int64_t *stride_,
                                                             const int64_t *on_stride_, const size_t off_set_,
                                                             const size_t output_size_, half *output,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<float>(const float *input, const int64_t *crops_start,
                                                              const int64_t *block_shape, const int64_t *output_shape,
                                                              const size_t output_shape_size, const int64_t *stride_,
                                                              const int64_t *on_stride_, const size_t off_set_,
                                                              const size_t output_size_, float *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpaceND<double>(const double *input, const int64_t *crops_start,
                                                               const int64_t *block_shape, const int64_t *output_shape,
                                                               const size_t output_shape_size, const int64_t *stride_,
                                                               const int64_t *on_stride_, const size_t off_set_,
                                                               const size_t output_size_, double *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
