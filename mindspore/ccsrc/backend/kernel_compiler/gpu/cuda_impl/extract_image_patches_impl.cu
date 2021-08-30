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

#include "backend/kernel_compiler/gpu/cuda_impl/extract_image_patches_impl.cuh"

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

template void CalExtractImagePatchesNHWC<int>(size_t output_size, int64_t stride_row, int64_t stride_col,
                                              int64_t rate_row, int64_t rate_col, int64_t output_cols, bool need_batch,
                                              int64_t row_stride, int64_t patch_stride, int64_t other_stride,
                                              int64_t input_row_size, int64_t input_col_size, int64_t row_padding_top,
                                              int64_t col_padding_left, int64_t col_input_stride,
                                              int64_t row_input_stride, int64_t patch_input_stride,
                                              int64_t output_depth, const int *input, int *output, cudaStream_t stream);
template void CalExtractImagePatchesNHWC<float>(size_t output_size, int64_t stride_row, int64_t stride_col,
                                                int64_t rate_row, int64_t rate_col, int64_t output_cols,
                                                bool need_batch, int64_t row_stride, int64_t patch_stride,
                                                int64_t other_stride, int64_t input_row_size, int64_t input_col_size,
                                                int64_t row_padding_top, int64_t col_padding_left,
                                                int64_t col_input_stride, int64_t row_input_stride,
                                                int64_t patch_input_stride, int64_t output_depth, const float *input,
                                                float *output, cudaStream_t stream);
template void CalExtractImagePatchesNHWC<half>(size_t output_size, int64_t stride_row, int64_t stride_col,
                                               int64_t rate_row, int64_t rate_col, int64_t output_cols, bool need_batch,
                                               int64_t row_stride, int64_t patch_stride, int64_t other_stride,
                                               int64_t input_row_size, int64_t input_col_size, int64_t row_padding_top,
                                               int64_t col_padding_left, int64_t col_input_stride,
                                               int64_t row_input_stride, int64_t patch_input_stride,
                                               int64_t output_depth, const half *input, half *output,
                                               cudaStream_t stream);
template void CalExtractImagePatchesNHWC<double>(size_t output_size, int64_t stride_row, int64_t stride_col,
                                                 int64_t rate_row, int64_t rate_col, int64_t output_cols,
                                                 bool need_batch, int64_t row_stride, int64_t patch_stride,
                                                 int64_t other_stride, int64_t input_row_size, int64_t input_col_size,
                                                 int64_t row_padding_top, int64_t col_padding_left,
                                                 int64_t col_input_stride, int64_t row_input_stride,
                                                 int64_t patch_input_stride, int64_t output_depth, const double *input,
                                                 double *output, cudaStream_t stream);
