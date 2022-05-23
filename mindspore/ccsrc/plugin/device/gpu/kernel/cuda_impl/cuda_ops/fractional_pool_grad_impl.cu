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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_pool_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void InitOutput(T *output, const int64_t outer_size) {
    T zero = 0;
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < outer_size; id += blockDim.x * gridDim.x) {
        output[id] = zero;
    }
    return;
}

template <typename T>
__global__ void Fractionalmaxpoolgrad(const T *orig_input, const T *orig_output, const T *out_backprop,
                                  const int64_t *row_pooling_sequence, const int64_t *col_pooling_sequence, T *output,
                                  const bool overlapping, const int64_t outputHeight, const int64_t outputWidth,
                                  const int64_t outputChannel, const int64_t backpropHeight,
                                  const int64_t backpropWidth, const int64_t backpropChannel,
                                  const int64_t backprop_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < backprop_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (backpropHeight * backpropWidth * backpropChannel);
    const int posh = pos / (backpropWidth * backpropChannel) % backpropHeight;
    const int posw = pos / backpropChannel % backpropWidth;
    const int posc = pos % backpropChannel;
    const int64_t height_start = row_pooling_sequence[posh];
    int64_t height_end = overlapping ? row_pooling_sequence[posh + 1] : row_pooling_sequence[posh + 1] - 1;
    const int64_t width_start = col_pooling_sequence[posw];
    int64_t width_end = overlapping ? col_pooling_sequence[posw + 1] : col_pooling_sequence[posw + 1] - 1;

    const int64_t height_max = outputHeight - 1;
    const int64_t width_max = outputWidth - 1;

    height_end = height_end < height_max ? height_end : height_max;
    width_end = width_end < width_max ? width_end : width_max;
    int max_in_offset = ((posn * outputHeight + height_start) * outputWidth + width_start) * outputChannel + posc;
    T max = orig_input[max_in_offset];
    for (int64_t h = height_start; h <= height_end; ++h) {
      for (int64_t w = width_start; w <= width_end; ++w) {
        const int64_t in_offset = ((posn * outputHeight + h) * outputWidth + w) * outputChannel + posc;
        if (max < orig_input[in_offset]) {
          max = orig_input[in_offset];
          max_in_offset = in_offset;
        }
      }
    }
    MsAtomicAdd(output + max_in_offset, out_backprop[pos]);
  }
  return;
}

template <typename T>
__global__ void Fractionalavgpoolgrad(const int64_t *orig_input, const T *out_backprop,
                                  const int64_t *row_pooling_sequence, const int64_t *col_pooling_sequence, T *output,
                                  const bool overlapping, const int64_t outputHeight, const int64_t outputWidth,
                                  const int64_t outputChannel, const int64_t backpropHeight,
                                  const int64_t backpropWidth, const int64_t backpropChannel,
                                  const int64_t backprop_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < backprop_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (backpropHeight * backpropWidth * backpropChannel);
    const int posh = pos / (backpropWidth * backpropChannel) % backpropHeight;
    const int posw = pos / backpropChannel % backpropWidth;
    const int posc = pos % backpropChannel;
    const int64_t height_start = row_pooling_sequence[posh];
    int64_t height_end = overlapping ? row_pooling_sequence[posh + 1] : row_pooling_sequence[posh + 1] - 1;
    const int64_t width_start = col_pooling_sequence[posw];
    int64_t width_end = overlapping ? col_pooling_sequence[posw + 1] : col_pooling_sequence[posw + 1] - 1;

    const int64_t height_max = outputHeight - 1;
    const int64_t width_max = outputWidth - 1;

    height_end = height_end < height_max ? height_end : height_max;
    width_end = width_end < width_max ? width_end : width_max;
    const int64_t num_elements_in_pooling_cell = (height_end - height_start + 1) * (width_end - width_start + 1);
    for (int64_t h = height_start; h <= height_end; ++h) {
      for (int64_t w = width_start; w <= width_end; ++w) {
        const int64_t out_offset = ((posn * outputHeight + h) * outputWidth + w) * outputChannel + posc;
        MsAtomicAdd(output + out_offset, (out_backprop[pos] / static_cast<T>(num_elements_in_pooling_cell)));
      }
    }
  }
  return;
}

template <typename T>
void CalFractionalmaxpoolgrad(const T *orig_input, const T *orig_output, const T *out_backprop,
                              const int64_t *row_pooling_sequence, const int64_t *col_pooling_sequence, T *output,
                              const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
                              const bool overlapping, const int64_t backprop_size, const int64_t outer_size,
                              const uint32_t &device_id, cudaStream_t cuda_stream) {
  InitOutput<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(output, outer_size);
  Fractionalmaxpoolgrad<<<CUDA_BLOCKS(device_id, backprop_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence, output, overlapping,
    output_shape[1], output_shape[2], output_shape[3], out_backprop_shape[1], out_backprop_shape[2],
    out_backprop_shape[3], backprop_size);
  return;
}

template <typename T>
void CalFractionalavgpoolgrad(const int64_t *orig_input, const T *out_backprop,
                              const int64_t *row_pooling_sequence, const int64_t *col_pooling_sequence, T *output,
                              const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
                              const bool overlapping, const int64_t backprop_size, const int64_t outer_size,
                              const uint32_t &device_id, cudaStream_t cuda_stream) {
  InitOutput<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(output, outer_size);
  Fractionalavgpoolgrad<<<CUDA_BLOCKS(device_id, backprop_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    orig_input, out_backprop, row_pooling_sequence, col_pooling_sequence, output, overlapping,
    output_shape[1], output_shape[2], output_shape[3], out_backprop_shape[1], out_backprop_shape[2],
    out_backprop_shape[3], backprop_size);
  return;
}

template CUDA_LIB_EXPORT void CalFractionalmaxpoolgrad<float>(const float *orig_input, const float *orig_output,
                                                              const float *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, float *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgrad<double>(const double *orig_input, const double *orig_output,
                                                              const double *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, double *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgrad<int32_t>(const int32_t *orig_input, const int32_t *orig_output,
                                                              const int32_t *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, int32_t *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgrad<int64_t>(const int64_t *orig_input, const int64_t *orig_output,
                                                              const int64_t *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, int64_t *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpoolgrad<float>(const int64_t *orig_input,
                                                              const float *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, float *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpoolgrad<double>(const int64_t *orig_input,
                                                              const double *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, double *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpoolgrad<int32_t>(const int64_t *orig_input,
                                                              const int32_t *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, int32_t *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalavgpoolgrad<int64_t>(const int64_t *orig_input,
                                                              const int64_t *out_backprop,
                                                              const int64_t *row_pooling_sequence,
                                                              const int64_t *col_pooling_sequence, int64_t *output,
                                                              const std::vector<int64_t> &out_backprop_shape,
                                                              const std::vector<int64_t> &output_shape,
                                                              const bool overlapping, const int64_t backprop_size,
                                                              const int64_t outer_size, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
