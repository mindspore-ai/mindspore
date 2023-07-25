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

#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dilation2d_backprop_filter_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void InitOutput(T *output, const int64_t outer_size) {
  T zero = 0;
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < outer_size; id += blockDim.x * gridDim.x) {
    output[id] = zero;
  }
  return;
}

template <typename T>
__global__ void Dilation2DBackpropFilter(const T *input, const T *filter, const T *out_backprop, T *output,
                                         const int64_t inputHeight, const int64_t inputWidth, const int64_t Channel,
                                         const int64_t filterHeight, const int64_t filterWidth,
                                         const int64_t outputHeight, const int64_t outputWidth,
                                         const int64_t strideHeight, const int64_t strideWidth,
                                         const int64_t rateHeight, const int64_t rateWidth, const int64_t pad_top,
                                         const int64_t pad_left, const int64_t outer_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (outputHeight * outputWidth * Channel);
    const int posc = pos / (outputHeight * outputWidth) % Channel;
    const int posh = pos / outputWidth % outputHeight;
    const int posw = pos % outputWidth;
    const int height_start = posh * strideHeight - pad_top;
    const int width_start = posw * strideWidth - pad_left;

    T max_val = std::numeric_limits<T>::lowest();
    int max_h = 0;
    int max_w = 0;

    for (int h = 0; h < filterHeight; ++h) {
      const int h_in = height_start + h * rateHeight;
      if (h_in >= 0 && h_in < inputHeight) {
        for (int w = 0; w < filterWidth; ++w) {
          const int w_in = width_start + w * rateWidth;
          if (w_in >= 0 && w_in < inputWidth) {
            const T val = input[w_in + inputWidth * (h_in + inputHeight * (posc + Channel * posn))] +
                          filter[w + filterWidth * (h + filterHeight * posc)];
            if (val > max_val) {
              max_val = val;
              max_h = h;
              max_w = w;
            }
          }
        }
      }
    }
    MsAtomicAdd(output + max_w + filterWidth * (max_h + filterHeight * posc), out_backprop[pos]);
  }
  return;
}

template <>
__global__ void Dilation2DBackpropFilter(const half *input, const half *filter, const half *out_backprop, half *output,
                                         const int64_t inputHeight, const int64_t inputWidth, const int64_t Channel,
                                         const int64_t filterHeight, const int64_t filterWidth,
                                         const int64_t outputHeight, const int64_t outputWidth,
                                         const int64_t strideHeight, const int64_t strideWidth,
                                         const int64_t rateHeight, const int64_t rateWidth, const int64_t pad_top,
                                         const int64_t pad_left, const int64_t outer_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (outputHeight * outputWidth * Channel);
    const int posc = pos / (outputHeight * outputWidth) % Channel;
    const int posh = pos / outputWidth % outputHeight;
    const int posw = pos % outputWidth;
    const int height_start = posh * strideHeight - pad_top;
    const int width_start = posw * strideWidth - pad_left;

    half max_val = __int2half_rd(-65504);
    int max_h = 0;
    int max_w = 0;

    for (int h = 0; h < filterHeight; ++h) {
      const int h_in = height_start + h * rateHeight;
      if (h_in >= 0 && h_in < inputHeight) {
        for (int w = 0; w < filterWidth; ++w) {
          const int w_in = width_start + w * rateWidth;
          if (w_in >= 0 && w_in < inputWidth) {
            const half val = input[w_in + inputWidth * (h_in + inputHeight * (posc + Channel * posn))] +
                             filter[w + filterWidth * (h + filterHeight * posc)];
            if (val > max_val) {
              max_val = val;
              max_h = h;
              max_w = w;
            }
          }
        }
      }
    }
    MsAtomicAdd(output + max_w + filterWidth * (max_h + filterHeight * posc), out_backprop[pos]);
  }
  return;
}

template <typename T>
cudaError_t CalDilation2DBackpropFilter(const T *input, const T *filter, const T *out_backprop, T *output,
                                        const std::vector<int64_t> &input_shape,
                                        const std::vector<int64_t> &filter_shape,
                                        const std::vector<int64_t> &out_backprop_shape,
                                        const std::vector<int64_t> &output_shape, const std::vector<int64_t> &stride,
                                        const std::vector<int64_t> &dilation, int64_t (&pads)[2],
                                        const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream) {
  InitOutput<<<CUDA_BLOCKS(device_id, output_shape[0] * output_shape[1] * output_shape[2]), CUDA_THREADS(device_id), 0,
               cuda_stream>>>(output, output_shape[0] * output_shape[1] * output_shape[2]);
  Dilation2DBackpropFilter<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, filter, out_backprop, output, input_shape[2], input_shape[3], input_shape[1], filter_shape[1],
    filter_shape[2], out_backprop_shape[2], out_backprop_shape[3], stride[2], stride[3], dilation[2], dilation[3],
    pads[0], pads[1], outer_size);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<half>(
  const half *input, const half *filter, const half *out_backprop, half *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<float>(
  const float *input, const float *filter, const float *out_backprop, float *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<double>(
  const double *input, const double *filter, const double *out_backprop, double *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<int32_t>(
  const int32_t *input, const int32_t *filter, const int32_t *out_backprop, int32_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<int64_t>(
  const int64_t *input, const int64_t *filter, const int64_t *out_backprop, int64_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<int8_t>(
  const int8_t *input, const int8_t *filter, const int8_t *out_backprop, int8_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<int16_t>(
  const int16_t *input, const int16_t *filter, const int16_t *out_backprop, int16_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<uint8_t>(
  const uint8_t *input, const uint8_t *filter, const uint8_t *out_backprop, uint8_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<uint16_t>(
  const uint16_t *input, const uint16_t *filter, const uint16_t *out_backprop, uint16_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<uint32_t>(
  const uint32_t *input, const uint32_t *filter, const uint32_t *out_backprop, uint32_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDilation2DBackpropFilter<uint64_t>(
  const uint64_t *input, const uint64_t *filter, const uint64_t *out_backprop, uint64_t *output,
  const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
  const std::vector<int64_t> &out_backprop_shape, const std::vector<int64_t> &output_shape,
  const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t (&pads)[2],
  const int64_t outer_size, const uint32_t &device_id, cudaStream_t cuda_stream);
