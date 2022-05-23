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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_pool_impl.cuh"

template <typename T>
__global__ void Fractionalmaxpool(const T *input, T *output, const int64_t *row_pooling_sequence,
                                  const int64_t *col_pooling_sequence, const bool overlapping,
                                  const int64_t inputHeight, const int64_t inputWidth,
                                  const int64_t inputChannel, const int64_t outputHeight,
                                  const int64_t outputWidth, const int64_t outputChannel, const int64_t outer_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (outputHeight * outputWidth * outputChannel);
    const int posh = pos / (outputWidth * outputChannel) % outputHeight;
    const int posw = pos / outputChannel % outputWidth;
    const int posc = pos % outputChannel;
    const int height_start = row_pooling_sequence[posh];
    int height_end = overlapping ? row_pooling_sequence[posh + 1] : (row_pooling_sequence[posh + 1] - 1);
    const int width_start = col_pooling_sequence[posw];
    int width_end = overlapping ? col_pooling_sequence[posw + 1] : (col_pooling_sequence[posw + 1] - 1);

    const int height_max = inputHeight - 1;
    const int width_max = inputWidth - 1;

    height_end = height_end < height_max ? height_end : height_max;
    width_end = width_end < width_max ? width_end : width_max;

    const int init_offset = ((posn * inputHeight + height_start) * inputWidth + width_start) * inputChannel + posc;
    T max = input[init_offset];
    for (int h = height_start; h <= height_end; ++h) {
      for (int w = width_start; w <= width_end; ++w) {
        const int in_offset = ((posn * inputHeight + h) * inputWidth + w) * inputChannel + posc;
        max = max > input[in_offset] ? max : input[in_offset];
      }
    }

    output[pos] = max;
  }
  return;
}

template <typename T>
__global__ void Fractionalavgpool(const T *input, T *output, const int64_t *row_pooling_sequence,
                                  const int64_t *col_pooling_sequence, const bool overlapping,
                                  const int64_t inputHeight, const int64_t inputWidth,
                                  const int64_t inputChannel, const int64_t outputHeight,
                                  const int64_t outputWidth, const int64_t outputChannel, const int64_t outer_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (outputHeight * outputWidth * outputChannel);
    const int posh = pos / (outputWidth * outputChannel) % outputHeight;
    const int posw = pos / outputChannel % outputWidth;
    const int posc = pos % outputChannel;
    const int height_start = row_pooling_sequence[posh];
    int height_end = overlapping ? row_pooling_sequence[posh + 1] : (row_pooling_sequence[posh + 1] - 1);
    const int width_start = col_pooling_sequence[posw];
    int width_end = overlapping ? col_pooling_sequence[posw + 1] : (col_pooling_sequence[posw + 1] - 1);
    const int height_max = inputHeight - 1;
    const int width_max = inputWidth - 1;

    height_end = height_end < height_max ? height_end : height_max;
    width_end = width_end < width_max ? width_end : width_max;

    T sum = static_cast<T>(0);
    int count = 0;
    for (int h = height_start; h <= height_end; ++h) {
      for (int w = width_start; w <= width_end; ++w) {
        const int in_offset = ((posn * inputHeight + h) * inputWidth + w) * inputChannel + posc;
        sum += input[in_offset];
        count++;
      }
    }
    T avg = sum / static_cast<T>(count);
    output[pos] = avg;
  }
  return;
}

template <typename T>
void CalFractionalmaxpool(const T *input, T *output, const int64_t *row_pooling_sequence,
                          const int64_t *col_pooling_sequence, const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &output_shape, const bool overlapping,
                          const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream) {
  Fractionalmaxpool<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, output, row_pooling_sequence, col_pooling_sequence, overlapping, input_shape[1], input_shape[2],
    input_shape[3], output_shape[1], output_shape[2], output_shape[3], outer_size);
  return;
}

template <typename T>
void CalFractionalavgpool(const T *input, T *output, const int64_t *row_pooling_sequence,
                          const int64_t *col_pooling_sequence, const std::vector<int64_t> &input_shape,
                          const std::vector<int64_t> &output_shape, const bool overlapping,
                          const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream) {
  Fractionalavgpool<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, output, row_pooling_sequence, col_pooling_sequence, overlapping, input_shape[1], input_shape[2],
    input_shape[3], output_shape[1], output_shape[2], output_shape[3], outer_size);
  return;
}

template CUDA_LIB_EXPORT void CalFractionalmaxpool<float>(const float *input, float *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpool<double>(const double *input, double *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpool<int32_t>(const int32_t *input, int32_t *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpool<int64_t>(const int64_t *input, int64_t *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpool<float>(const float *input, float *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpool<double>(const double *input, double *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpool<int32_t>(const int32_t *input, int32_t *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpool<int64_t>(const int64_t *input, int64_t *output,
                                                          const int64_t *row_pooling_sequence,
                                                          const int64_t *col_pooling_sequence,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &output_shape,
                                                          const bool overlapping,
                                                          const int64_t outer_size,
                                                          const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
