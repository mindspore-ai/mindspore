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
__global__ void MaxPool3DWithArgmax(const T *input, T *output, S *index, const int inputD, const int inputH,
                                    const int inputW, const int ksizeD, const int ksizeH, const int ksizeW,
                                    const int strideD, const int strideH, const int strideW, const int padF,
                                    const int padT, const int padL, const int dilationD, const int dilationH,
                                    const int dilationW, const int outD, const int outH, const int outW,
                                    const int offset) {
  int posd = (blockIdx.z + offset) % outD;
  int posh = blockIdx.y * blockDim.y + threadIdx.y;
  int posw = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (blockIdx.z + offset) / outD;

  if (posh < outH && posw < outW) {
    int dstart = posd * strideD - padF;
    int hstart = posh * strideH - padT;
    int wstart = posw * strideW - padL;
    int dend = min(dstart + (ksizeD - 1) * dilationD + 1, inputD);
    int hend = min(hstart + (ksizeH - 1) * dilationH + 1, inputH);
    int wend = min(wstart + (ksizeW - 1) * dilationW + 1, inputW);
    while (dstart < 0) {
      dstart += dilationD;
    }
    while (hstart < 0) {
      hstart += dilationH;
    }
    while (wstart < 0) {
      wstart += dilationW;
    }

    int inHW = inputH * inputW;
    input += stride * inputD * inputH * inputW;
    S maxIdx = dstart * inHW + hstart * inputW + wstart;
    T maxValue = input[maxIdx];
    for (int dcur = dstart; dcur < dend; dcur += dilationD) {
      for (int hcur = hstart; hcur < hend; hcur += dilationH) {
        for (int wcur = wstart; wcur < wend; wcur += dilationW) {
          S input_idx = dcur * inHW + hcur * inputW + wcur;
          T value = input[input_idx];
          if ((value > maxValue) || IsNan(value)) {
            maxIdx = input_idx;
            maxValue = value;
          }
        }
      }
    }
    int pos = stride * outD * outH * outW + posd * outH * outW + posh * outW + posw;
    output[pos] = maxValue;
    index[pos] = maxIdx;
  }
}

template <typename T, typename S>
cudaError_t CalMaxPool3DWithArgmax(const T *input, const int n, const int c, const int d, const int h, const int w,
                                   const int ksizeD, const int ksizeH, const int ksizeW, const int strideD,
                                   const int strideH, const int strideW, const int padF, const int padT, const int padL,
                                   const int dilationD, const int dilationH, const int dilationW, const int outD,
                                   const int outH, const int outW, T *output, S *index, const uint32_t device_id,
                                   cudaStream_t cuda_stream) {
  int size = n * c * outD;
  int offset = 0;
  dim3 block(32, 8);
  dim3 grid = CUDA_GRIDS_MAXSIZE(device_id);
  grid.x = (outW + static_cast<int>(block.x) - 1) / static_cast<int>(block.x);
  grid.y = (outH + static_cast<int>(block.y) - 1) / static_cast<int>(block.y);
  int max_grid_z = static_cast<int>(grid.z);
  while (size > 0) {
    grid.z = size > max_grid_z ? max_grid_z : size;
    MaxPool3DWithArgmax<<<grid, block, 0, cuda_stream>>>(input, output, index, d, h, w, ksizeD, ksizeH, ksizeW, strideD,
                                                         strideH, strideW, padF, padT, padL, dilationD, dilationH,
                                                         dilationW, outD, outH, outW, offset);
    size -= max_grid_z;
    offset += max_grid_z;
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<half, int32_t>(
  const half *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, half *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<float, int32_t>(
  const float *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, float *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<double, int32_t>(
  const double *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, double *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int8_t, int32_t>(
  const int8_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int8_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int16_t, int32_t>(
  const int16_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int16_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int32_t, int32_t>(
  const int32_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int32_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int64_t, int32_t>(
  const int64_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int64_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint8_t, int32_t>(
  const uint8_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint8_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint16_t, int32_t>(
  const uint16_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint16_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint32_t, int32_t>(
  const uint32_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint32_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint64_t, int32_t>(
  const uint64_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint64_t *output, int32_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<half, int64_t>(
  const half *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, half *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<float, int64_t>(
  const float *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, float *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<double, int64_t>(
  const double *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, double *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int8_t, int64_t>(
  const int8_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int8_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int16_t, int64_t>(
  const int16_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int16_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int32_t, int64_t>(
  const int32_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int32_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<int64_t, int64_t>(
  const int64_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, int64_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint8_t, int64_t>(
  const uint8_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint8_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint16_t, int64_t>(
  const uint16_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint16_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint32_t, int64_t>(
  const uint32_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint32_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalMaxPool3DWithArgmax<uint64_t, int64_t>(
  const uint64_t *input, const int n, const int c, const int d, const int h, const int w, const int ksizeD,
  const int ksizeH, const int ksizeW, const int strideD, const int strideH, const int strideW, const int padF,
  const int padT, const int padL, const int dilationD, const int dilationH, const int dilationW, const int outD,
  const int outH, const int outW, uint64_t *output, int64_t *index, const uint32_t device_id, cudaStream_t cuda_stream);
