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

#include "resize_bicubic_impl.cuh"
#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

__device__ int Bound(int access, int limit) {
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
__global__ void ResizeBicubic(const T *input, const float A, const int n, const int c, const int input_h,
                              const int input_w, const int output_h, const int output_w, const int nchw, const int chw,
                              const int hw, const float h_scale, const float w_scale, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += gridDim.x * blockDim.x) {
    int posn = pos / chw;
    int posc = pos / hw % c;
    int posh = pos / output_w % output_h;
    int posw = pos % output_w;
    float posw_scaled = 0;
    float posh_scaled = 0;
    posw_scaled = w_scale * posw;
    posh_scaled = h_scale * posh;
    const int w_low = static_cast<int>(floorf(posw_scaled));
    const int h_low = static_cast<int>(floorf(posh_scaled));
    const float w_alpha = posw_scaled - w_low;
    const float h_alpha = posh_scaled - h_low;
    float coefficients[4];
    float coeffs0, coeffs1, coeffs2, coeffs3;
    float temp0, temp1, temp2, temp3;
    int input_start;
    int access_h;
    int access_w;
    const int64_t offset_w = lrintf(w_alpha * 1024);
    float x = offset_w * 1.0 / 1024;
    x += 1.0;
    coeffs0 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    x -= 1.0;
    coeffs1 = ((A + 2) * x - (A + 3)) * x * x + 1;
    x = (1024 - offset_w) * 1.0 / 1024;
    coeffs2 = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    coeffs3 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    for (int k = 0; k < 4; k++) {
      access_h = Bound(h_low - 1 + k, input_h);
      access_w = Bound(w_low - 1, input_w);
      input_start = input_w * input_h * (c * posn + posc) + access_h * input_w;
      temp0 = input[input_start + access_w];
      access_w = Bound(w_low, input_w);
      temp1 = input[input_start + access_w];
      access_w = Bound(w_low + 1, input_w);
      temp2 = input[input_start + access_w];
      access_w = Bound(w_low + 2, input_w);
      temp3 = input[input_start + access_w];
      coefficients[k] = coeffs0 * temp0 + coeffs1 * temp1 + coeffs2 * temp2 + coeffs3 * temp3;
    }
    const int64_t offset_h = lrintf(h_alpha * 1024);
    x = offset_h * 1.0 / 1024;
    x += 1.0;
    coeffs0 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    x -= 1.0;
    coeffs1 = ((A + 2) * x - (A + 3)) * x * x + 1;
    x = (1024 - offset_h) * 1.0 / 1024;
    coeffs2 = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    coeffs3 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    S result = static_cast<S>(coefficients[0] * coeffs0 + coefficients[1] * coeffs1 + coefficients[2] * coeffs2 +
                              coefficients[3] * coeffs3);
    output[pos] = result;
  }
  return;
}

template <typename T, typename S>
__global__ void ResizeBicubicHalfPixelCenters(const T *input, const float A, const int n, const int c,
                                              const int input_h, const int input_w, const int output_h,
                                              const int output_w, const int nchw, const int chw, const int hw,
                                              const float h_scale, const float w_scale, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += gridDim.x * blockDim.x) {
    int posn = pos / chw;
    int posc = pos / hw % c;
    int posh = pos / output_w % output_h;
    int posw = pos % output_w;
    float posw_scaled = 0;
    float posh_scaled = 0;
    posw_scaled = (static_cast<float>(posw) + static_cast<float>(0.5)) * w_scale - static_cast<float>(0.5);
    posh_scaled = (static_cast<float>(posh) + static_cast<float>(0.5)) * h_scale - static_cast<float>(0.5);
    const int w_low = static_cast<int>(floorf(posw_scaled));
    const int h_low = static_cast<int>(floorf(posh_scaled));
    const float w_alpha = posw_scaled - w_low;
    const float h_alpha = posh_scaled - h_low;
    float coefficients[4];
    float coeffs0, coeffs1, coeffs2, coeffs3;
    float temp0, temp1, temp2, temp3;
    int input_start;
    int access_h;
    int access_w;
    const int64_t offset_w = lrintf(w_alpha * 1024);
    float x = offset_w * 1.0 / 1024;
    access_w = Bound(w_low - 1, input_w);
    x += 1.0;
    temp0 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    coeffs0 = (access_w == w_low - 1 ? temp0 : 0.0f);
    access_w = Bound(w_low, input_w);
    x -= 1.0;
    temp1 = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs1 = (access_w == w_low ? temp1 : 0.0f);
    access_w = Bound(w_low + 1, input_w);
    x = (1024 - offset_w) * 1.0 / 1024;
    temp2 = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs2 = (access_w == w_low + 1 ? temp2 : 0.0f);
    access_w = Bound(w_low + 2, input_w);
    x += 1.0;
    temp3 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    coeffs3 = (access_w == w_low + 2 ? temp3 : 0.0f);
    float sumcoeff = coeffs0 + coeffs1 + coeffs2 + coeffs3;
    if (std::abs(sumcoeff) >= 1000.0f * std::numeric_limits<float>::min()) {
      const float one_over_weight_sum = 1.0f / sumcoeff;
      coeffs0 *= one_over_weight_sum;
      coeffs1 *= one_over_weight_sum;
      coeffs2 *= one_over_weight_sum;
      coeffs3 *= one_over_weight_sum;
    }
    for (int k = 0; k < 4; k++) {
      access_h = Bound(h_low - 1 + k, input_h);
      access_w = Bound(w_low - 1, input_w);
      input_start = input_w * input_h * (c * posn + posc) + access_h * input_w;
      temp0 = input[input_start + access_w];
      access_w = Bound(w_low, input_w);
      temp1 = input[input_start + access_w];
      access_w = Bound(w_low + 1, input_w);
      temp2 = input[input_start + access_w];
      access_w = Bound(w_low + 2, input_w);
      temp3 = input[input_start + access_w];
      coefficients[k] = coeffs0 * temp0 + coeffs1 * temp1 + coeffs2 * temp2 + coeffs3 * temp3;
    }
    const int64_t offset_h = lrintf(h_alpha * 1024);
    x = offset_h * 1.0 / 1024;
    access_h = Bound(h_low - 1, input_h);
    x += 1.0;
    temp0 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    coeffs0 = (access_h == h_low - 1 ? temp0 : 0.0f);
    access_h = Bound(h_low, input_h);
    x -= 1.0;
    temp1 = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs1 = (access_h == h_low ? temp1 : 0.0f);
    access_h = Bound(h_low + 1, input_h);
    x = (1024 - offset_h) * 1.0 / 1024;
    temp2 = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs2 = (access_h == h_low + 1 ? temp2 : 0.0f);
    access_h = Bound(h_low + 2, input_h);
    x += 1.0;
    temp3 = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    coeffs3 = (access_h == h_low + 2 ? temp3 : 0.0f);
    float sumcoeff2 = coeffs0 + coeffs1 + coeffs2 + coeffs3;
    if (std::abs(sumcoeff2) >= 1000.0f * std::numeric_limits<float>::min()) {
      const float one_over_weight_sum2 = 1.0f / sumcoeff2;
      coeffs0 *= one_over_weight_sum2;
      coeffs1 *= one_over_weight_sum2;
      coeffs2 *= one_over_weight_sum2;
      coeffs3 *= one_over_weight_sum2;
    }
    S result = static_cast<S>(coefficients[0] * coeffs0 + coefficients[1] * coeffs1 + coefficients[2] * coeffs2 +
                              coefficients[3] * coeffs3);
    output[pos] = result;
  }
  return;
}

template <typename T, typename S>
__global__ void ResizeBicubicSame(const T *input, S *output, int nchw) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += gridDim.x * blockDim.x) {
    S val = input[pos];
    output[pos] = val;
    return;
  }
}

template <typename T, typename S>
void CalResizeBicubic(const T *input, const int n, const int c, const int input_h, const int input_w,
                      const int output_h, const int output_w, const float h_scale, const float w_scale, S *output,
                      bool half_pixel_centers, const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int hw = output_h * output_w;
  const int chw = c * hw;
  const int nchw = n * chw;
  if (input_h == output_h && input_w == output_w) {
    ResizeBicubicSame<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(input, output, nchw);
    return;
  }
  float A = -0.75;
  if (half_pixel_centers == true) {
    A = -0.5;
    ResizeBicubicHalfPixelCenters<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, A, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output);
    return;
  } else {
    ResizeBicubic<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, A, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output);
    return;
  }
}

template CUDA_LIB_EXPORT void CalResizeBicubic<half, half>(const half *input, const int n, const int c,
                                                           const int input_h, const int input_w, const int output_h,
                                                           const int output_w, const float h_scale, const float w_scale,
                                                           half *output, bool half_pixel_centers,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalResizeBicubic<float, float>(const float *input, const int n, const int c,
                                                             const int input_h, const int input_w, const int output_h,
                                                             const int output_w, const float h_scale,
                                                             const float w_scale, float *output,
                                                             bool half_pixel_centers, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalResizeBicubic<double, double>(const double *input, const int n, const int c,
                                                               const int input_h, const int input_w, const int output_h,
                                                               const int output_w, const float h_scale,
                                                               const float w_scale, double *output,
                                                               bool half_pixel_centers, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
