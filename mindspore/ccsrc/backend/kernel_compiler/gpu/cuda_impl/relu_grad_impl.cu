/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/relu_grad_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void CalReLUGradKernel(int size, T *dy, T *y, T *dx) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    dx[pos] = y[pos] > static_cast<T>(0) ? dy[pos] : static_cast<T>(0);
  }
}

template <typename T>
void CalReLUGrad(int size, T *dy, T *y, T *dx, cudaStream_t cuda_stream) {
  CalReLUGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, y, dx);
  return;
}

template <typename T>
__global__ void PReluChannelSharedGradKernel(size_t size, T *dy_addr, T *x_addr, T *w_addr, T *dx_addr, T *dwc_addr) {
  T zero = static_cast<T>(0);
  T w = w_addr[0];
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T dy = dy_addr[pos];
    T x = x_addr[pos];
    dx_addr[pos] = x > zero ? dy : w * dy;
    dwc_addr[pos] = x > zero ? zero : x * dy;
  }
}

template <typename T>
void PReluChannelSharedGrad(size_t input_size, T *dy_addr, T *x_addr, T *w_addr, T *dx_addr, T *dwc_addr,
                            cudaStream_t cuda_stream) {
  PReluChannelSharedGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, dy_addr, x_addr,
                                                                                        w_addr, dx_addr, dwc_addr);
  return;
}

template void CalReLUGrad(int size, double *dy, double *y, double *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, float *dy, float *y, float *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, half *dy, half *y, half *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int8_t *dy, int8_t *y, int8_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int16_t *dy, int16_t *y, int16_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int32_t *dy, int32_t *y, int32_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int64_t *dy, int64_t *y, int64_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, uint8_t *dy, uint8_t *y, uint8_t *dx, cudaStream_t cuda_stream);
template void PReluChannelSharedGrad(size_t input_size, float *dy_addr, float *x_addr, float *w_addr, float *dx_addr,
                                     float *dwc_addr, cudaStream_t cuda_stream);
template void PReluChannelSharedGrad(size_t input_size, half *dy_addr, half *x_addr, half *w_addr, half *dx_addr,
                                     half *dwc_addr, cudaStream_t cuda_stream);
