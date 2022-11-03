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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/extract_image_patches_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void ExtractImagePatches(size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row,
                                    int64_t rate_col, int64_t output_cols, bool need_batch, int64_t row_stride,
                                    int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
                                    int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left,
                                    int64_t col_input_stride, int64_t row_input_stride, int64_t patch_input_stride,
                                    int64_t output_depth, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += blockDim.x * gridDim.x) {
    const int64_t batch_index = need_batch ? (static_cast<int64_t>(pos) / other_stride) : 0;
    const int64_t inner_index =
      need_batch ? (static_cast<int64_t>(pos) - batch_index * other_stride) : static_cast<int64_t>(pos);
    // inner index
    const int64_t patch_index = inner_index / patch_stride;
    const int64_t patch_offset = (inner_index - patch_index * patch_stride) / output_depth;
    // row
    const int64_t row_index = patch_index / output_cols;
    const int64_t row_offset = patch_offset / row_stride;
    const int64_t input_row = row_index * stride_row + row_offset * rate_row - row_padding_top;
    if (input_row < 0 || input_row >= input_row_size) {
      output[pos] = static_cast<T>(0);
      continue;
    }
    // col
    const int64_t col_index = patch_index - row_index * output_cols;
    const int64_t col_offset = patch_offset - row_offset * row_stride;
    const int64_t input_col = col_index * stride_col + col_offset * rate_col - col_padding_left;
    if (input_col < 0 || input_col >= input_col_size) {
      output[pos] = static_cast<T>(0);
      continue;
    }
    // depth
    const int64_t depth = inner_index - (inner_index / output_depth) * output_depth;
    // input index
    const int64_t input_index =
      depth + input_col * col_input_stride + input_row * row_input_stride + batch_index * patch_input_stride;
    output[pos] = input[static_cast<size_t>(input_index)];
  }
  return;
}

template <typename T>
void CalExtractImagePatchesNHWC(size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row,
                                int64_t rate_col, int64_t output_cols, bool need_batch, int64_t row_stride,
                                int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
                                int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left,
                                int64_t col_input_stride, int64_t row_input_stride, int64_t patch_input_stride,
                                int64_t output_depth, const T *input, T *output, cudaStream_t stream) {
  ExtractImagePatches<<<GET_BLOCKS(output_size), GET_THREADS, 0, stream>>>(
    output_size, stride_row, stride_col, rate_row, rate_col, output_cols, need_batch, row_stride, patch_stride,
    other_stride, input_row_size, input_col_size, row_padding_top, col_padding_left, col_input_stride, row_input_stride,
    patch_input_stride, output_depth, input, output);
}

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<half>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const half *input, half *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<float>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const float *input, float *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<double>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const double *input, double *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<int8_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const int8_t *input, int8_t *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<int16_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const int16_t *input, int16_t *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<int32_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const int32_t *input, int32_t *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<int64_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const int64_t *input, int64_t *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<uint8_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const uint8_t *input, uint8_t *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<uint16_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const uint16_t *input, uint16_t *output,
  cudaStream_t stream);
template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<uint32_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const uint32_t *input, uint32_t *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<uint64_t>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const uint64_t *input, uint64_t *output,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<Complex<float>>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const Complex<float> *input,
  Complex<float> *output, cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<Complex<double>>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const Complex<double> *input,
  Complex<double> *output, cudaStream_t stream);

template CUDA_LIB_EXPORT void CalExtractImagePatchesNHWC<bool>(
  size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row, int64_t rate_col, int64_t output_cols,
  bool need_batch, int64_t row_stride, int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
  int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left, int64_t col_input_stride,
  int64_t row_input_stride, int64_t patch_input_stride, int64_t output_depth, const bool *input, bool *output,
  cudaStream_t stream);
