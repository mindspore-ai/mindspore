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
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

template <typename T, typename S>
__global__ void MaxPoolWithArgmaxGrad(const T* dy,
                                      const S* index,
                                      const int xCHW,
                                      const int dyCHW,
                                      const int dyNCHW,
                                      T* dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (dyNCHW); pos += blockDim.x * gridDim.x) {
    const S idx = index[pos];
    const int posn = pos / dyCHW;
    MsAtomicAdd(dx + posn*xCHW + static_cast<int>(idx), dy[pos]);
  }
  return;
}

template <typename T>
__global__ void InitOutput(const int size, T *output) {
    T zero = 0;
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
        output[id] = zero;
    }
    return;
}

template <typename T, typename S>
void CalMaxPoolWithArgmaxGrad(const T* dy,
                              const S* index,
                              const int n,
                              const int c,
                              const int xHeight,
                              const int xWidth,
                              const int dyHeight,
                              const int dyWidth,
                              T* dx,
                              cudaStream_t cuda_stream) {
  const int xHW = xHeight*xWidth;
  const int xCHW = c*xHW;
  const int xNCHW = n*xCHW;
  const int dyHW = dyHeight*dyWidth;
  const int dyCHW = c*dyHW;
  const int dyNCHW = n*dyCHW;
  InitOutput<<<GET_BLOCKS(xNCHW), GET_THREADS, 0, cuda_stream>>>(xNCHW, dx);
  MaxPoolWithArgmaxGrad<<<GET_BLOCKS(dyNCHW),
                          GET_THREADS,
                          0,
                          cuda_stream>>>(
                            dy,
                            index,
                            xCHW,
                            dyCHW,
                            dyNCHW,
                            dx);
  return;
}

template void CalMaxPoolWithArgmaxGrad<float, int>(const float* dy,
                                                    const int* index,
                                                    const int n,
                                                    const int c,
                                                    const int xHeight,
                                                    const int xWidth,
                                                    const int dyHeight,
                                                    const int dyWidth,
                                                    float* dx,
                                                    cudaStream_t cuda_stream);
template void CalMaxPoolWithArgmaxGrad<half, int>(const half* dy,
                                                    const int* index,
                                                    const int n,
                                                    const int c,
                                                    const int xHeight,
                                                    const int xWidth,
                                                    const int dyHeight,
                                                    const int dyWidth,
                                                    half* dx,
                                                    cudaStream_t cuda_stream);
