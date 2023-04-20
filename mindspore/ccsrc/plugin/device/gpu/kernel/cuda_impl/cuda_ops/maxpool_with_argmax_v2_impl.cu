/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include "maxpool_with_argmax_v2_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T, typename S>
__global__ void MaxPoolWithArgmaxV2(const T *input, T *output, S *index, const int inputN, const int inputC,
                                    const int inputH, const int inputW, const int ksizeH, const int ksizeW,
                                    const int stridesH, const int stridesW, const int padsH, const int padsW,
                                    const int dilationH, const int dilationW, const int outH, const int outW) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (inputN * inputC * outH * outW);
       pos += blockDim.x * gridDim.x) {
    const int pos_n = pos / (inputC * outH * outW);
    const int pos_c = pos / (outH * outW) % inputC;
    const int pos_h = pos / outW % outH;
    const int pos_w = pos % outW;
    int start_h = pos_h * stridesH - padsH;
    int start_w = pos_w * stridesW - padsW;
    const int end_h = min(start_h + (ksizeH - 1) * dilationH + 1, inputH);
    const int end_w = min(start_w + (ksizeW - 1) * dilationW + 1, inputW);
    start_h = max(start_h, 0);
    start_w = max(start_w, 0);
    S input_start = pos_n * inputC * inputH * inputW;
    S stride = pos_c * inputH * inputW;
    S max_idx = stride + start_h * inputW + start_w;
    T max_data = input[input_start + max_idx];
    for (int cur_h = start_h; cur_h < end_h; cur_h++) {
      for (int cur_w = start_w; cur_w < end_w; cur_w++) {
        S input_idx = stride + cur_h * inputW + cur_w;
        T input_data = input[input_start + input_idx];
        if (input_data > max_data) {
          max_idx = input_idx - stride;
          max_data = input_data;
        }
      }
    }
    output[pos] = max_data;
    index[pos] = max_idx;
  }
}

template <typename T, typename S>
void CalMaxPoolWithArgmaxV2(const T *input, const int n, const int c, const int h, const int w, const int ksize_h,
                            const int ksize_w, const int strides_h, const int strides_w, const int pads_h,
                            const int pads_w, const int dilation_h, const int dilation_w, const int out_h,
                            const int out_w, T *output, S *index, const uint32_t &device_id, cudaStream_t cuda_stream) {
  MaxPoolWithArgmaxV2<<<CUDA_BLOCKS(device_id, n * c * out_h * out_w), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, output, index, n, c, h, w, ksize_h, ksize_w, strides_h, strides_w, pads_h, pads_w, dilation_h, dilation_w,
    out_h, out_w);
}

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<half, int32_t>(
  const half *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, half *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<float, int32_t>(
  const float *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, float *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<double, int32_t>(
  const double *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, double *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int8_t, int32_t>(
  const int8_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int8_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int16_t, int32_t>(
  const int16_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int16_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int32_t, int32_t>(
  const int32_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int32_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int64_t, int32_t>(
  const int64_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int64_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint8_t, int32_t>(
  const uint8_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint8_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint16_t, int32_t>(
  const uint16_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint16_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint32_t, int32_t>(
  const uint32_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint32_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint64_t, int32_t>(
  const uint64_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint64_t *output, int32_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<half, int64_t>(
  const half *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, half *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<float, int64_t>(
  const float *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, float *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<double, int64_t>(
  const double *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, double *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int8_t, int64_t>(
  const int8_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int8_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int16_t, int64_t>(
  const int16_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int16_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int32_t, int64_t>(
  const int32_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int32_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<int64_t, int64_t>(
  const int64_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, int64_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint8_t, int64_t>(
  const uint8_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint8_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint16_t, int64_t>(
  const uint16_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint16_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint32_t, int64_t>(
  const uint32_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint32_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2<uint64_t, int64_t>(
  const uint64_t *input, const int n, const int c, const int h, const int w, const int ksize_h, const int ksize_w,
  const int strides_h, const int strides_w, const int pads_h, const int pads_w, const int dilation_h,
  const int dilation_w, const int out_h, const int out_w, uint64_t *output, int64_t *index, const uint32_t &device_id,
  cudaStream_t cuda_stream);
