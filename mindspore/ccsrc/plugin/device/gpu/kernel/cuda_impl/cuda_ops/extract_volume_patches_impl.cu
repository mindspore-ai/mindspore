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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/extract_volume_patches_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void ExtractVolumePatches(size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col,
                                     int64_t output_depth, int64_t output_height, int64_t output_width, bool need_batch,
                                     int64_t d_stride, int64_t h_stride, int64_t w_stride, int64_t patch_stride,
                                     int64_t other_stride, int64_t input_channel, int64_t input_dep_size,
                                     int64_t input_row_size, int64_t input_col_size, int64_t pad_head, int64_t pad_top,
                                     int64_t pad_left, int64_t chan_input_stride, int64_t dep_input_stride,
                                     int64_t row_input_stride, int64_t patch_input_stride, const T *input, T *output) {
  size_t pos;
  for (size_t w_pos = blockIdx.x * blockDim.x + threadIdx.x; w_pos < output_size / (w_stride * input_channel);
       w_pos += blockDim.x * gridDim.x) {
    pos = static_cast<size_t>(w_pos / patch_stride) * w_stride * input_channel * patch_stride + (w_pos % patch_stride);
    const int64_t batch_index = need_batch ? (static_cast<int64_t>(pos) / other_stride) : 0;
    const int64_t inner_index =
      need_batch ? (static_cast<int64_t>(pos) - batch_index * other_stride) : static_cast<int64_t>(pos);
    // inner index
    const int64_t patch_index = inner_index % patch_stride;
    const int64_t patch_offset = inner_index / patch_stride / input_channel;
    // channel
    const int64_t channel = inner_index / patch_stride % input_channel;
    // depth
    const int64_t dep_index = patch_index / (output_height * output_width);
    const int64_t dep_offset = patch_offset / d_stride;
    const int64_t input_dep = dep_index * stride_dep + dep_offset - pad_head;
    if (input_dep < 0 || input_dep >= input_dep_size) {
      continue;
    }
    // height
    const int64_t row_index = patch_index / output_width % output_height;
    const int64_t row_offset = patch_offset / w_stride % h_stride;
    const int64_t input_row = row_index * stride_row + row_offset - pad_top;
    if (input_row < 0 || input_row >= input_row_size) {
      continue;
    }
    // width
    const int64_t col_index = patch_index % output_width;
    const int64_t col_offset = patch_offset % w_stride;
    const int64_t input_col = col_index * stride_col + col_offset - pad_left;
    // input index
    const int64_t input_index = input_col + input_row * row_input_stride + input_dep * dep_input_stride +
                                channel * chan_input_stride + batch_index * patch_input_stride;
#pragma unroll
    for (int64_t i = 0; i < w_stride; i++) {
      if (input_col + i < 0) {
        continue;
      }
      if (input_col + i >= input_col_size) {
        break;
      }
#pragma unroll
      for (int64_t j = 0; j < input_channel; j++) {
        output[pos + (i * input_channel + j) * patch_stride] = input[input_index + i + j * chan_input_stride];
      }
    }
  }
  return;
}

template <typename T>
cudaError_t CalExtractVolumePatches(size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col,
                                    int64_t output_depth, int64_t output_height, int64_t output_width, bool need_batch,
                                    int64_t d_stride, int64_t h_stride, int64_t w_stride, int64_t patch_stride,
                                    int64_t other_stride, int64_t input_channel, int64_t input_dep_size,
                                    int64_t input_row_size, int64_t input_col_size, int64_t pad_head, int64_t pad_top,
                                    int64_t pad_left, int64_t chan_input_stride, int64_t dep_input_stride,
                                    int64_t row_input_stride, int64_t patch_input_stride, const T *input, T *output,
                                    cudaStream_t stream) {
  cudaMemsetAsync(output, 0, sizeof(T) * output_size, stream);
  ExtractVolumePatches<<<GET_BLOCKS(output_size / (w_stride * input_channel)), GET_THREADS, 0, stream>>>(
    output_size, stride_dep, stride_row, stride_col, output_depth, output_height, output_width, need_batch, d_stride,
    h_stride, w_stride, patch_stride, other_stride, input_channel, input_dep_size, input_row_size, input_col_size,
    pad_head, pad_top, pad_left, chan_input_stride, dep_input_stride, row_input_stride, patch_input_stride, input,
    output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<double>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const double *input, double *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<float>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const float *input, float *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<half>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const half *input, half *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<int64_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const int64_t *input, int64_t *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<int32_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const int32_t *input, int32_t *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<int16_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const int16_t *input, int16_t *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<int8_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const int8_t *input, int8_t *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<uint64_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const uint64_t *input,
  uint64_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<uint32_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const uint32_t *input,
  uint32_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<uint16_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const uint16_t *input,
  uint16_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalExtractVolumePatches<uint8_t>(
  size_t output_size, int64_t stride_dep, int64_t stride_row, int64_t stride_col, int64_t output_depth,
  int64_t output_height, int64_t output_width, bool need_batch, int64_t d_stride, int64_t h_stride, int64_t w_stride,
  int64_t patch_stride, int64_t other_stride, int64_t input_channel, int64_t input_dep_size, int64_t input_row_size,
  int64_t input_col_size, int64_t pad_head, int64_t pad_top, int64_t pad_left, int64_t chan_input_stride,
  int64_t dep_input_stride, int64_t row_input_stride, int64_t patch_input_stride, const uint8_t *input, uint8_t *output,
  cudaStream_t stream);
