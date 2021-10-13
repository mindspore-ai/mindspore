/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/resize_bilinear_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"
template <typename T>
__global__ void ResizeBilinear(const T *input, const int n, const int c, const int input_h, const int input_w,
  const int output_h, const int output_w, const int nchw, const int chw, const int hw, const float h_scale,
  const float w_scale, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / output_w % output_h;
    const int posw = pos % output_w;
    const float posw_scaled = w_scale * posw;
    const float posh_scaled = h_scale * posh;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0); // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), input_w - 1); // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0); // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), input_h - 1); // NOLINT
    const float w_alpha = posw_scaled - w_low;
    const float w_beta = 1.0f - w_alpha;
    const float h_alpha = posh_scaled - h_low;
    const float h_beta = 1.0f - h_alpha;
    const int input_start = input_h * input_w * (posn * c  + posc);
    const T p1 = input[input_start + (h_low * input_w) + w_low];
    const T p2 = input[input_start + (h_low * input_w) + w_high];
    const T p3 = input[input_start + (h_high * input_w) + w_low];
    const T p4 = input[input_start + (h_high * input_w) + w_high];
    output[pos] = (p1 * static_cast<T>(h_beta * w_beta)) + (p2 * static_cast<T>(h_beta * w_alpha))
                  + (p3 * static_cast<T>(h_alpha * w_beta)) + (p4 * static_cast<T>(h_alpha * w_alpha));
  }
  return;
}

// fp16 path
__global__ void ResizeBilinearGrad(const half *input, const int n, const int c, const int input_h, const int input_w,
  const int output_h, const int output_w, const int nchw, const int chw, const int hw, const float h_scale,
  const float w_scale, half *output, float *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / input_w % input_h;
    const int posw = pos % input_w;
    const float posw_scaled = w_scale * posw;
    const float posh_scaled = h_scale * posh;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0); // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), output_w - 1); // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0); // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), output_h - 1); // NOLINT
    const float w_alpha = posw_scaled - w_low;
    const float w_beta = 1.0f - w_alpha;
    const float h_alpha = posh_scaled - h_low;
    const float h_beta = 1.0f - h_alpha;
    const float grad = static_cast<float>(input[pos]);
    const float dp1 = h_beta * w_beta * grad;
    const float dp2 = h_beta * w_alpha * grad;
    const float dp3 = h_alpha * w_beta * grad;
    const float dp4 = h_alpha * w_alpha * grad;
    const int output_start = output_h * output_w * (posn * c  + posc);
    atomicAdd(&interim[output_start + (h_low * output_w) + w_low], dp1);
    atomicAdd(&interim[output_start + (h_low * output_w) + w_high], dp2);
    atomicAdd(&interim[output_start + (h_high * output_w) + w_low], dp3);
    atomicAdd(&interim[output_start + (h_high * output_w) + w_high], dp4);
  }
  return;
}

// fp32 path
__global__ void ResizeBilinearGrad(const float *input, const int n, const int c, const int input_h, const int input_w,
  const int output_h, const int output_w, const int nchw, const int chw, const int hw, const float h_scale,
  const float w_scale, float *output, float *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / input_w % input_h;
    const int posw = pos % input_w;
    const float posw_scaled = w_scale * posw;
    const float posh_scaled = h_scale * posh;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0); // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), output_w - 1); // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0); // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), output_h - 1); // NOLINT
    const float w_alpha = posw_scaled - w_low;
    const float w_beta = 1.0f - w_alpha;
    const float h_alpha = posh_scaled - h_low;
    const float h_beta = 1.0f - h_alpha;
    const float grad = input[pos];
    const float dp1 = h_beta * w_beta * grad;
    const float dp2 = h_beta * w_alpha * grad;
    const float dp3 = h_alpha * w_beta * grad;
    const float dp4 = h_alpha * w_alpha * grad;
    const int output_start = output_h * output_w * (posn * c  + posc);
    atomicAdd(&output[output_start + (h_low * output_w) + w_low], dp1);
    atomicAdd(&output[output_start + (h_low * output_w) + w_high], dp2);
    atomicAdd(&output[output_start + (h_high * output_w) + w_low], dp3);
    atomicAdd(&output[output_start + (h_high * output_w) + w_high], dp4);
  }
  return;
}

__global__ void ResizeBilinearGradPost(const int nchw, half *output, float *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    output[pos] = __float2half(interim[pos]);
  }
  return;
}

template <typename T>
void CalResizeBilinear(const T *input, const int n, const int c, const int input_h, const int input_w,
  const int output_h, const int output_w, const float h_scale, const float w_scale, T *output,
  cudaStream_t cuda_stream) {
  const int nchw = n * c * output_h * output_w;
  const int chw = c * output_h * output_w;
  const int hw = output_h * output_w;
  ResizeBilinear<<<GET_BLOCKS(nchw), GET_THREADS, 0, cuda_stream>>>(input, n, c, input_h, input_w, output_h,
    output_w, nchw, chw, hw, h_scale, w_scale, output);
  return;
}

void CalResizeBilinearGrad(const half *input, const int n, const int c, const int input_h, const int input_w,
  const int output_h, const int output_w, const float h_scale, const float w_scale, half *output, float *interim,
  cudaStream_t cuda_stream) {
  const int hw = input_h * input_w;
  const int chw = c * hw;
  const int nchw = n * chw;
  const int output_num = n * c * output_h * output_w;
  ResizeBilinearGrad<<<GET_BLOCKS(nchw), GET_THREADS, 0, cuda_stream>>>(input, n, c, input_h, input_w, output_h,
    output_w, nchw, chw, hw, h_scale, w_scale, output, interim);
  ResizeBilinearGradPost<<<GET_BLOCKS(output_num), GET_THREADS, 0, cuda_stream>>>(output_num, output, interim);
  return;
}

void CalResizeBilinearGrad(const float *input, const int n, const int c, const int input_h, const int input_w,
  const int output_h, const int output_w, const float h_scale, const float w_scale, float *output, float *interim,
  cudaStream_t cuda_stream) {
  const int hw = input_h * input_w;
  const int chw = c * hw;
  const int nchw = n * chw;
  ResizeBilinearGrad<<<GET_BLOCKS(nchw), GET_THREADS, 0, cuda_stream>>>(input, n, c, input_h, input_w, output_h,
    output_w, nchw, chw, hw, h_scale, w_scale, output, interim);
  return;
}

template void CalResizeBilinear<float>(const float *input, const int n, const int c, const int input_h,
  const int input_w, const int output_h, const int output_w, const float h_scale, const float w_scale, float *output,
  cudaStream_t cuda_stream);
template void CalResizeBilinear<half>(const half *input, const int n, const int c, const int input_h,
  const int input_w, const int output_h, const int output_w, const float h_scale, const float w_scale, half *output,
  cudaStream_t cuda_stream);
