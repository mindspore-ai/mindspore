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

#include "resize_bicubic_grad_impl.cuh"
#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

__device__ int Bounds(int access, int limit) {
  int access_value = access;
  if (access_value < 0) {
    access_value = 0;
  }
  if (access_value > limit - 1) {
    access_value = limit - 1;
  }
  return access_value;
}

template <typename T, typename S>
__global__ void ResizeBicubicGradSame(const T *input, S *output, int nchw) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += gridDim.x * blockDim.x) {
    S val = input[pos];
    output[pos] = val;
    return;
  }
}

template <typename T, typename S>
__global__ void ResizeBicubicGrad(const T *input, const S A, const int n, const int c, const int grad_h,
                                  const int grad_w, const int origin_h, const int origin_w, const int nchw,
                                  const int chw, const int hw, const float h_scale, const float w_scale, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += gridDim.x * blockDim.x) {
    int posn = pos / chw;
    int posc = pos / hw % c;
    int posh = pos / grad_w % grad_h;
    int posw = pos % grad_w;
    S posw_scaled = 0;
    S posh_scaled = 0;
    posw_scaled = w_scale * posw;
    posh_scaled = h_scale * posh;
    const int w_low = static_cast<int>(floorf(posw_scaled));
    const int h_low = static_cast<int>(floorf(posh_scaled));
    const S w_alpha = posw_scaled - w_low;
    const S h_alpha = posh_scaled - h_low;
    S x_coeffs[4];
    S y_coeffs[4];
    int temp;
    int input_start;
    int access_h;
    int access_w;
    const int64_t offset_w = lrintf(w_alpha * 1024);
    S x = offset_w * 1.0 / 1024;
    const int64_t offset_h = lrintf(h_alpha * 1024);
    S y = offset_h * 1.0 / 1024;
    x += 1.0;
    x_coeffs[0] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    x -= 1.0;
    x_coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x = (1024 - offset_w) * 1.0 / 1024;
    x_coeffs[2] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    x_coeffs[3] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    y += 1.0;
    y_coeffs[0] = ((A * y - 5 * A) * y + 8 * A) * y - 4 * A;
    y -= 1.0;
    y_coeffs[1] = ((A + 2) * y - (A + 3)) * y * y + 1;
    y = (1024 - offset_h) * 1.0 / 1024;
    y_coeffs[2] = ((A + 2) * y - (A + 3)) * y * y + 1;
    y += 1.0;
    y_coeffs[3] = ((A * y - 5 * A) * y + 8 * A) * y - 4 * A;
    S value = static_cast<S>(input[pos]);
    for (int k = 0; k < 4; k++) {
      for (int m = 0; m < 4; m++) {
        access_h = Bounds(h_low - 1 + k, origin_h);
        access_w = Bounds(w_low - 1 + m, origin_w);
        input_start = origin_w * origin_h * (c * posn + posc) + access_h * origin_w;
        temp = input_start + access_w;
        MsAtomicAdd(&output[temp], value * y_coeffs[k] * x_coeffs[m]);
      }
    }
  }
  return;
}

template <typename T, typename S>
__global__ void ResizeBicubicGradHalfPixelCenters(const T *input, const S A, const int n, const int c, const int grad_h,
                                                  const int grad_w, const int origin_h, const int origin_w,
                                                  const int nchw, const int chw, const int hw, const float h_scale,
                                                  const float w_scale, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += gridDim.x * blockDim.x) {
    int posn = pos / chw;
    int posc = pos / hw % c;
    int posh = pos / grad_w % grad_h;
    int posw = pos % grad_w;
    S posw_scaled = 0;
    S posh_scaled = 0;
    posw_scaled = (static_cast<S>(posw) + static_cast<S>(0.5)) * w_scale - static_cast<S>(0.5);
    posh_scaled = (static_cast<S>(posh) + static_cast<S>(0.5)) * h_scale - static_cast<S>(0.5);
    const int w_low = static_cast<int>(floorf(posw_scaled));
    const int h_low = static_cast<int>(floorf(posh_scaled));
    const S w_alpha = posw_scaled - w_low;
    const S h_alpha = posh_scaled - h_low;
    S x_coeffs[4];
    S y_coeffs[4];
    S temp0, temp1, temp2, temp3;
    int temp;
    int input_start;
    int access_h;
    int access_w;
    const int64_t offset_w = lrintf(w_alpha * 1024);
    S x = offset_w * 1.0 / 1024;
    const int64_t offset_h = lrintf(h_alpha * 1024);
    S y = offset_h * 1.0 / 1024;
    access_w = Bounds(w_low - 1, origin_w);
    x += 1.0;
    temp0 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    x_coeffs[0] = (access_w == w_low - 1 ? temp0 : 0.0f);
    access_w = Bounds(w_low, origin_w);
    x -= 1.0;
    temp1 = ((A + 2) * x - (A + 3)) * x * x + 1;
    x_coeffs[1] = (access_w == w_low ? temp1 : 0.0f);
    access_w = Bounds(w_low + 1, origin_w);
    x = (1024 - offset_w) * 1.0 / 1024;
    temp2 = ((A + 2) * x - (A + 3)) * x * x + 1;
    x_coeffs[2] = (access_w == w_low + 1 ? temp2 : 0.0f);
    access_w = Bounds(w_low + 2, origin_w);
    x += 1.0;
    temp3 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    x_coeffs[3] = (access_w == w_low + 2 ? temp3 : 0.0f);
    S sumcoeff = x_coeffs[0] + x_coeffs[1] + x_coeffs[2] + x_coeffs[3];
    if (std::abs(sumcoeff) >= 1000.0f * std::numeric_limits<float>::min()) {
      const S one_over_weight_sum = 1.0f / sumcoeff;
      x_coeffs[0] *= one_over_weight_sum;
      x_coeffs[1] *= one_over_weight_sum;
      x_coeffs[2] *= one_over_weight_sum;
      x_coeffs[3] *= one_over_weight_sum;
    }
    access_h = Bounds(h_low - 1, origin_h);
    y += 1.0;
    temp0 = ((A * y - 5 * A) * y + 8 * A) * y - 4 * A;
    y_coeffs[0] = (access_h == h_low - 1 ? temp0 : 0.0f);
    access_h = Bounds(h_low, origin_h);
    y -= 1.0;
    temp1 = ((A + 2) * y - (A + 3)) * y * y + 1;
    y_coeffs[1] = (access_h == h_low ? temp1 : 0.0f);
    access_h = Bounds(h_low + 1, origin_h);
    y = (1024 - offset_h) * 1.0 / 1024;
    temp2 = ((A + 2) * y - (A + 3)) * y * y + 1;
    y_coeffs[2] = (access_h == h_low + 1 ? temp2 : 0.0f);
    access_h = Bounds(h_low + 2, origin_h);
    y += 1.0;
    temp3 = ((A * y - 5 * A) * y + 8 * A) * y - 4 * A;
    y_coeffs[3] = (access_h == h_low + 2 ? temp3 : 0.0f);
    S sumcoeff2 = y_coeffs[0] + y_coeffs[1] + y_coeffs[2] + y_coeffs[3];
    if (std::abs(sumcoeff2) >= 1000.0f * std::numeric_limits<float>::min()) {
      const S one_over_weight_sum2 = 1.0f / sumcoeff2;
      y_coeffs[0] *= one_over_weight_sum2;
      y_coeffs[1] *= one_over_weight_sum2;
      y_coeffs[2] *= one_over_weight_sum2;
      y_coeffs[3] *= one_over_weight_sum2;
    }
    S value = static_cast<S>(input[pos]);
    for (int k = 0; k < 4; k++) {
      for (int m = 0; m < 4; m++) {
        access_h = Bounds(h_low - 1 + k, origin_h);
        access_w = Bounds(w_low - 1 + m, origin_w);
        input_start = origin_w * origin_h * (c * posn + posc) + access_h * origin_w;
        temp = input_start + access_w;
        MsAtomicAdd(&output[temp], value * y_coeffs[k] * x_coeffs[m]);
      }
    }
  }
  return;
}

template <typename T, typename S>
void CalResizeBicubicGrad(const T *input, const int n, const int c, const int grad_h, const int grad_w,
                          const int origin_h, const int origin_w, const float h_scale, const float w_scale, S *output,
                          bool half_pixel_centers, const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int hw = grad_w * grad_h;
  const int chw = c * hw;
  const int nchw = n * chw;
  const int origin_size = n * c * origin_h * origin_w;
  cudaMemset(static_cast<void *>(output), 0, sizeof(S) * origin_size);
  if (origin_h == grad_h && origin_w == grad_w) {
    ResizeBicubicGradSame<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(input, output,
                                                                                                     nchw);
    return;
  }
  S A = -0.75;
  if (half_pixel_centers == true) {
    A = -0.5;
    ResizeBicubicGradHalfPixelCenters<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, A, n, c, grad_h, grad_w, origin_h, origin_w, nchw, chw, hw, h_scale, w_scale, output);
    return;
  } else {
    ResizeBicubicGrad<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, A, n, c, grad_h, grad_w, origin_h, origin_w, nchw, chw, hw, h_scale, w_scale, output);
    return;
  }
}

template CUDA_LIB_EXPORT void CalResizeBicubicGrad<float, float>(const float *input, const int n, const int c,
                                                                 const int grad_h, const int grad_w, const int origin_h,
                                                                 const int origin_w, const float h_scale,
                                                                 const float w_scale, float *output,
                                                                 bool half_pixel_centers, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalResizeBicubicGrad<double, double>(const double *input, const int n, const int c,
                                                                   const int grad_h, const int grad_w,
                                                                   const int origin_h, const int origin_w,
                                                                   const float h_scale, const float w_scale,
                                                                   double *output, bool half_pixel_centers,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
