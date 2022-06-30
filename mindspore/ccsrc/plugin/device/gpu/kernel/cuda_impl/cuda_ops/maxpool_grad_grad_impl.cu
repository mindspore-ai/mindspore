/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_grad_grad_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void MaxPoolGradGrad(const T *input, const T *grad, const int n, const int c, const int h, const int w,
                                const int windowHeight, const int windowWidth, const int strideHeight,
                                const int strideWidth, const int padTop, const int padLeft, const int outputHeight,
                                const int outputWidth, const int outputNCHW, const int outputCHW, const int outputHW,
                                T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (outputNCHW); pos += blockDim.x * gridDim.x) {
    const int posn = pos / outputCHW;
    const int posc = pos / outputHW % c;
    const int posh = pos / outputWidth % outputHeight;
    const int posw = pos % outputWidth;
    int hstart = posh * strideHeight - padTop;
    int wstart = posw * strideWidth - padLeft;
    const int hend = min(hstart + windowHeight, h);
    const int wend = min(wstart + windowWidth, w);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    int inputStart = posn * c * h * w + posc * h * w;
    int maxIdx = hstart * w + wstart;
    T maxData = input[inputStart + maxIdx];
    for (int hcur = hstart; hcur < hend; ++hcur) {
      for (int wcur = wstart; wcur < wend; ++wcur) {
        int inputIdx = hcur * w + wcur;
        T inputData = input[inputStart + inputIdx];
        if (inputData > maxData) {
          maxIdx = inputIdx;
          maxData = inputData;
        }
      }
    }
    output[pos] = grad[inputStart + maxIdx];
  }
}

template <typename T>
void CalMaxPoolGradGrad(const T *input, const T *grad, const int n, const int c, const int h, const int w,
                        const int windowHeight, const int windowWidth, const int strideHeight, const int strideWidth,
                        const int padTop, const int padLeft, const int outputHeight, const int outputWidth, T *output,
                        const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int outputNCHW = n * c * outputHeight * outputWidth;
  const int outputCHW = c * outputHeight * outputWidth;
  const int outputHW = outputHeight * outputWidth;
  MaxPoolGradGrad<<<CUDA_BLOCKS(device_id, n * c * outputHeight * outputWidth), CUDA_THREADS(device_id), 0,
                    cuda_stream>>>(input, grad, n, c, h, w, windowHeight, windowWidth, strideHeight, strideWidth,
                                   padTop, padLeft, outputHeight, outputWidth, outputNCHW, outputCHW, outputHW, output);
}

template CUDA_LIB_EXPORT void CalMaxPoolGradGrad<float>(const float *input, const float *grad, const int n, const int c,
                                                        const int h, const int w, const int windowHeight,
                                                        const int windowWidth, const int strideHeight,
                                                        const int strideWidth, const int padTop, const int padLeft,
                                                        const int outputHeight, const int outputWidth, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolGradGrad<half>(const half *input, const half *grad, const int n, const int c,
                                                       const int h, const int w, const int windowHeight,
                                                       const int windowWidth, const int strideHeight,
                                                       const int strideWidth, const int padTop, const int padLeft,
                                                       const int outputHeight, const int outputWidth, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
__global__ void MaxPool3DGradGrad(const T *input, const T *grad, const int n, const int c, const int d, const int h,
                                  const int w, const int windowDepth, const int windowHeight, const int windowWidth,
                                  const int strideDepth, const int strideHeight, const int strideWidth,
                                  const int padFront, const int padTop, const int padLeft, const int outputDepth,
                                  const int outputHeight, const int outputWidth, T *output) {
  const int outputNCDHW = n * c * outputDepth * outputHeight * outputWidth;
  const int outputCDHW = c * outputDepth * outputHeight * outputWidth;
  const int outputDHW = outputDepth * outputHeight * outputWidth;
  const int outputHW = outputHeight * outputWidth;

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (outputNCDHW); pos += blockDim.x * gridDim.x) {
    const int posn = pos / outputCDHW;
    const int posc = pos / outputDHW % c;
    const int posd = pos / outputHW % outputDepth;
    const int posh = pos / outputWidth % outputHeight;
    const int posw = pos % outputWidth;

    int dstart = posd * strideDepth - padFront;
    int hstart = posh * strideHeight - padTop;
    int wstart = posw * strideWidth - padLeft;
    const int dend = min(dstart + windowDepth, d);
    const int hend = min(hstart + windowHeight, h);
    const int wend = min(wstart + windowWidth, w);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    int inputStart = posn * c * d * h * w + posc * d * h * w;
    int maxIdx = dstart * h * w + hstart * w + wstart;
    T maxData = input[inputStart + maxIdx];
    for (int dcur = dstart; dcur < dend; ++dcur) {
      for (int hcur = hstart; hcur < hend; ++hcur) {
        for (int wcur = wstart; wcur < wend; ++wcur) {
          int inputIdx = dcur * h * w + hcur * w + wcur;
          T inputData = input[inputStart + inputIdx];
          if (inputData > maxData) {
            maxIdx = inputIdx;
            maxData = inputData;
          }
        }
      }
    }
    output[pos] = grad[inputStart + maxIdx];
  }
}

template <typename T>
void CalMaxPool3DGradGrad(const T *input, const T *grad, const int n, const int c, const int d, const int h,
                          const int w, const int windowDepth, const int windowHeight, const int windowWidth,
                          const int strideDepth, const int strideHeight, const int strideWidth, const int padFront,
                          const int padTop, const int padLeft, const int outputDepth, const int outputHeight,
                          const int outputWidth, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  MaxPool3DGradGrad<<<CUDA_BLOCKS(device_id, n * c * outputDepth * outputHeight * outputWidth), CUDA_THREADS(device_id),
                      0, cuda_stream>>>(input, grad, n, c, d, h, w, windowDepth, windowHeight, windowWidth, strideDepth,
                                        strideHeight, strideWidth, padFront, padTop, padLeft, outputDepth, outputHeight,
                                        outputWidth, output);
}

template CUDA_LIB_EXPORT void CalMaxPool3DGradGrad<float>(
  const float *input, const float *grad, const int n, const int c, const int d, const int h, const int w,
  const int windowDepth, const int windowHeight, const int windowWidth, const int strideDepth, const int strideHeight,
  const int strideWidth, const int padFront, const int padTop, const int padLeft, const int outputDepth,
  const int outputHeight, const int outputWidth, float *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DGradGrad<half>(
  const half *input, const half *grad, const int n, const int c, const int d, const int h, const int w,
  const int windowDepth, const int windowHeight, const int windowWidth, const int strideDepth, const int strideHeight,
  const int strideWidth, const int padFront, const int padTop, const int padLeft, const int outputDepth,
  const int outputHeight, const int outputWidth, half *output, const uint32_t &device_id, cudaStream_t cuda_stream);
