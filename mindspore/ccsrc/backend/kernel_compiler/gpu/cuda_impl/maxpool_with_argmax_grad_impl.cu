/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "maxpool_with_argmax_grad_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"

template <typename T, typename S>
__global__ void MaxPoolWithArgmaxGrad(const T* x,
                                      const T* dy,
                                      const S* index,
                                      const int n,
                                      const int c,
                                      const int xHeight,
                                      const int xWidth,
                                      const int dyHeight,
                                      const int dyWidth,
                                      const int windowHeight,
                                      const int windowWidth,
                                      const int strideHeight,
                                      const int strideWidth,
                                      const int padTop,
                                      const int padLeft,
                                      const int xNCHW,
                                      const int xCHW,
                                      const int xHW,
                                      const int dyCHW,
                                      const int dyHW,
                                      T* dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
       pos < (xNCHW);
       pos += blockDim.x * gridDim.x) {
    const int posn = pos / xCHW;
    const int posc = pos / xHW % c;
    const int posh = pos / xHeight % xHeight;
    const int posw = pos % xWidth;
    const S posIdx = posh*xWidth + posw;
    int hstart = posh+padTop;
    if (hstart < windowHeight) {
      hstart = 0;
    } else {
      hstart = (hstart-windowHeight)/strideHeight + 1;
    }
    int wstart = posw+padLeft;
    if (wstart < windowWidth) {
      wstart = 0;
    } else {
      wstart = (wstart-windowWidth)/strideWidth + 1;
    }
    const int hend = min((posh+padTop)/strideHeight +1, dyHeight);
    const int wend = min((posw+padLeft)/strideWidth +1, dyWidth);
    const int channelStart = posn*dyCHW + posc*dyHW;
    T dySum = static_cast<T>(0.0);
    for (int hcur = hstart; hcur < hend; ++hcur) {
      for (int wcur = wstart; wcur < wend; ++wcur) {
        const int curIdx = hcur*dyWidth + wcur;
        S maxIdx = index[channelStart+curIdx];
        if (maxIdx == posIdx) {
          dySum += dy[channelStart+curIdx];
        }
      }
    }
    dx[pos] = dySum;
  }
  return;
}

template <>
__global__ void MaxPoolWithArgmaxGrad(const half* x,
                                      const half* dy,
                                      const int* index,
                                      const int n,
                                      const int c,
                                      const int xHeight,
                                      const int xWidth,
                                      const int dyHeight,
                                      const int dyWidth,
                                      const int windowHeight,
                                      const int windowWidth,
                                      const int strideHeight,
                                      const int strideWidth,
                                      const int padTop,
                                      const int padLeft,
                                      const int xNCHW,
                                      const int xCHW,
                                      const int xHW,
                                      const int dyCHW,
                                      const int dyHW,
                                      half* dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
       pos < (xNCHW);
       pos += blockDim.x * gridDim.x) {
    const int posn = pos / xCHW;
    const int posc = pos / xHW % c;
    const int posh = pos / xHeight % xHeight;
    const int posw = pos % xWidth;
    const int posIdx = posh*xWidth + posw;
    int hstart = posh+padTop;
    if (hstart < windowHeight) {
      hstart = 0;
    } else {
      hstart = (hstart-windowHeight)/strideHeight + 1;
    }
    int wstart = posw+padLeft;
    if (wstart < windowWidth) {
      wstart = 0;
    } else {
      wstart = (wstart-windowWidth)/strideWidth + 1;
    }
    const int hend = min((posh+padTop)/strideHeight +1, dyHeight);
    const int wend = min((posw+padLeft)/strideWidth +1, dyWidth);
    const int channelStart = posn*dyCHW + posc*dyHW;
    float dySum = 0.0f;
    for (int hcur = hstart; hcur < hend; ++hcur) {
      for (int wcur = wstart; wcur < wend; ++wcur) {
        const int curIdx = hcur*dyWidth + wcur;
        int maxIdx = index[channelStart+curIdx];
        if (maxIdx == posIdx) {
          dySum += __half2float(dy[channelStart+curIdx]);
        }
      }
    }
    dx[pos] = __float2half(dySum);
  }
  return;
}

template <typename T, typename S>
void CalMaxPoolWithArgmaxGrad(const T* x,
                              const T* dy,
                              const S* index,
                              const int n,
                              const int c,
                              const int xHeight,
                              const int xWidth,
                              const int dyHeight,
                              const int dyWidth,
                              const int windowHeight,
                              const int windowWidth,
                              const int strideHeight,
                              const int strideWidth,
                              const int padTop,
                              const int padLeft,
                              T* dx,
                              cudaStream_t cuda_stream) {
  const int xHW = xHeight*xWidth;
  const int xCHW = c*xHW;
  const int xNCHW = n*xCHW;
  const int dyHW = dyHeight*dyWidth;
  const int dyCHW = c*dyHW;
  MaxPoolWithArgmaxGrad<<<GET_BLOCKS(xNCHW),
                          GET_THREADS,
                          0,
                          cuda_stream>>>(
                            x,
                            dy,
                            index,
                            n,
                            c,
                            xHeight,
                            xWidth,
                            dyHeight,
                            dyWidth,
                            windowHeight,
                            windowWidth,
                            strideHeight,
                            strideWidth,
                            padTop,
                            padLeft,
                            xNCHW,
                            xCHW,
                            xHW,
                            dyCHW,
                            dyHW,
                            dx);
  return;
}

template void CalMaxPoolWithArgmaxGrad<float, int>(const float* x,
                                                    const float* dy,
                                                    const int* index,
                                                    const int n,
                                                    const int c,
                                                    const int xHeight,
                                                    const int xWidth,
                                                    const int dyHeight,
                                                    const int dyWidth,
                                                    const int windowHeight,
                                                    const int windowWidth,
                                                    const int strideHeight,
                                                    const int strideWidth,
                                                    const int padTop,
                                                    const int padLeft,
                                                    float* dx,
                                                    cudaStream_t cuda_stream);
template void CalMaxPoolWithArgmaxGrad<half, int>(const half* x,
                                                    const half* dy,
                                                    const int* index,
                                                    const int n,
                                                    const int c,
                                                    const int xHeight,
                                                    const int xWidth,
                                                    const int dyHeight,
                                                    const int dyWidth,
                                                    const int windowHeight,
                                                    const int windowWidth,
                                                    const int strideHeight,
                                                    const int strideWidth,
                                                    const int padTop,
                                                    const int padLeft,
                                                    half* dx,
                                                    cudaStream_t cuda_stream);
