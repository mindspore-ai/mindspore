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

#include <algorithm>
#include "maxpool3d_with_argmax_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ inline bool IsNan(T x) {
  return isnan(x);
}

__device__ inline bool IsNan(half x) { return __hisnan(x); }

template <typename T, typename S>
__global__ void MaxPool3DWithArgmax(const T *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h,
                                    const int64_t w, const int64_t ksize_depth, const int64_t ksize_height,
                                    const int64_t ksize_width, const int64_t stride_depth, const int64_t stride_height,
                                    const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
                                    const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height,
                                    const int64_t dilation_width, const int64_t out_depth, const int64_t out_height,
                                    const int64_t out_width, const int64_t out_ncdhw, const int64_t out_cdhw,
                                    const int64_t out_dhw, const int64_t out_hw, T *output, S *index) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (out_ncdhw); pos += blockDim.x * gridDim.x) {
    int64_t posn = pos / out_cdhw;
    int64_t posc = pos / out_dhw % c;
    int64_t posd = pos / out_hw % out_depth;
    int64_t posh = pos / out_width % out_height;
    int64_t posw = pos % out_width;

    int64_t dstart = posd * stride_depth - pad_front;
    int64_t hstart = posh * stride_height - pad_top;
    int64_t wstart = posw * stride_width - pad_left;
    int64_t dend = min(dstart + (ksize_depth - 1) * dilation_depth + 1, d);
    int64_t hend = min(hstart + (ksize_height - 1) * dilation_height + 1, h);
    int64_t wend = min(wstart + (ksize_width - 1) * dilation_width + 1, w);
    while (dstart < 0) {
      dstart += dilation_depth;
    }
    while (hstart < 0) {
      hstart += dilation_height;
    }
    while (wstart < 0) {
      wstart += dilation_width;
    }

    int64_t dhw_size = d * h * w;
    int64_t hw_size = h * w;
    S inputStart = (posn * c + posc) * dhw_size;
    S maxIdx = dstart * hw_size + hstart * w + wstart;
    T maxData = input[inputStart + maxIdx];
    for (int64_t dcur = dstart; dcur < dend; dcur += dilation_depth) {
      for (int64_t hcur = hstart; hcur < hend; hcur += dilation_height) {
        for (int64_t wcur = wstart; wcur < wend; wcur += dilation_width) {
          S inputIdx = dcur * hw_size + hcur * w + wcur;
          T inputData = input[inputStart + inputIdx];
          if ((inputData > maxData) || IsNan(inputData)) {
            maxIdx = inputIdx;
            maxData = inputData;
          }
        }
      }
    }
    output[pos] = maxData;
    index[pos] = maxIdx;
  }
}

template <typename T, typename S>
void CalMaxPool3DWithArgmax(const T *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h,
                            const int64_t w, const int64_t ksize_depth, const int64_t ksize_height,
                            const int64_t ksize_width, const int64_t stride_depth, const int64_t stride_height,
                            const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
                            const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height,
                            const int64_t dilation_width, const int64_t out_depth, const int64_t out_height,
                            const int64_t out_width, T *output, S *index, const uint32_t device_id,
                            cudaStream_t cuda_stream) {
  const int64_t out_ncdhw = n * c * out_depth * out_height * out_width;
  const int64_t out_cdhw = c * out_depth * out_height * out_width;
  const int64_t out_dhw = out_depth * out_height * out_width;
  const int64_t out_hw = out_height * out_width;

  MaxPool3DWithArgmax<<<CUDA_BLOCKS(device_id, out_ncdhw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, n, c, d, h, w, ksize_depth, ksize_height, ksize_width, stride_depth, stride_height, stride_width, pad_front,
    pad_top, pad_left, dilation_depth, dilation_height, dilation_width, out_depth, out_height, out_width, out_ncdhw,
    out_cdhw, out_dhw, out_hw, output, index);
}

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<half, int32_t>(
  const half *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, half *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<float, int32_t>(
  const float *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, float *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<double, int32_t>(
  const double *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, double *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int8_t, int32_t>(
  const int8_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int8_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int16_t, int32_t>(
  const int16_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int16_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int32_t, int32_t>(
  const int32_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int32_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int64_t, int32_t>(
  const int64_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int64_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint8_t, int32_t>(
  const uint8_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint8_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint16_t, int32_t>(
  const uint16_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint16_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint32_t, int32_t>(
  const uint32_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint32_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint64_t, int32_t>(
  const uint64_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint64_t *output, int32_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<half, int64_t>(
  const half *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, half *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<float, int64_t>(
  const float *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, float *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<double, int64_t>(
  const double *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, double *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int8_t, int64_t>(
  const int8_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int8_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int16_t, int64_t>(
  const int16_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int16_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int32_t, int64_t>(
  const int32_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int32_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<int64_t, int64_t>(
  const int64_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, int64_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint8_t, int64_t>(
  const uint8_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint8_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint16_t, int64_t>(
  const uint16_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint16_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint32_t, int64_t>(
  const uint32_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint32_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax<uint64_t, int64_t>(
  const uint64_t *input, const int64_t n, const int64_t c, const int64_t d, const int64_t h, const int64_t w,
  const int64_t ksize_depth, const int64_t ksize_height, const int64_t ksize_width, const int64_t stride_depth,
  const int64_t stride_height, const int64_t stride_width, const int64_t pad_front, const int64_t pad_top,
  const int64_t pad_left, const int64_t dilation_depth, const int64_t dilation_height, const int64_t dilation_width,
  const int64_t out_depth, const int64_t out_height, const int64_t out_width, uint64_t *output, int64_t *index,
  const uint32_t device_id, cudaStream_t cuda_stream);
