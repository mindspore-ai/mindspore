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

#include "adjusthue_impl.cuh"
#include <algorithm>
#include <cmath>

struct RgbTuple {
  float cu_r;
  float cu_g;
  float cu_b;
};

struct HsvTuple {
  float cu_h;
  float cu_s;
  float cu_v;
};

__device__ __forceinline__ HsvTuple rgb2hsv_cuda(const float cu_r, const float cu_g, const float cu_b) {
  HsvTuple tuple;
  const float cu_M = max(cu_r, max(cu_g, cu_b));
  const float cu_m = min(cu_r, min(cu_g, cu_b));
  const float cu_chroma = cu_M - cu_m;
  float cu_h = 0.0f;
  float cu_s = 0.0f;
  if (cu_chroma > 0.0f) {
    if (cu_M == cu_r) {
      const float cu_num = (cu_g - cu_b) / cu_chroma;
      const float cu_sign = copysignf(1.0f, cu_num);
      cu_h = ((cu_sign < 0.0f) * 6.0f + cu_sign * fmodf(cu_sign * cu_num, 6.0f)) / 6.0f;
    } else if (cu_M == cu_g) {
      cu_h = ((cu_b - cu_r) / cu_chroma + 2.0f) / 6.0f;
    } else {
      cu_h = ((cu_r - cu_g) / cu_chroma + 4.0f) / 6.0f;
    }
  } else {
    cu_h = 0.0f;
  }
  if (cu_M > 0.0f) {
    cu_s = cu_chroma / cu_M;
  } else {
    cu_s = 0.0f;
  }
  tuple.cu_h = cu_h;
  tuple.cu_s = cu_s;
  tuple.cu_v = cu_M;
  return tuple;
}

__device__ __forceinline__ RgbTuple hsv2rgb_cuda(const float cu_h, const float cu_s, const float cu_v) {
  RgbTuple tuple;
  const float cu_new_h = cu_h * 6.0f;
  const float cu_chroma = cu_v * cu_s;
  const float cu_x = cu_chroma * (1.0f - fabsf(fmodf(cu_new_h, 2.0f) - 1.0f));
  const float cu_new_m = cu_v - cu_chroma;
  const bool cu_between_0_and_1 = cu_new_h >= 0.0f && cu_new_h < 1.0f;
  const bool cu_between_1_and_2 = cu_new_h >= 1.0f && cu_new_h < 2.0f;
  const bool cu_between_2_and_3 = cu_new_h >= 2.0f && cu_new_h < 3.0f;
  const bool cu_between_3_and_4 = cu_new_h >= 3.0f && cu_new_h < 4.0f;
  const bool cu_between_4_and_5 = cu_new_h >= 4.0f && cu_new_h < 5.0f;
  const bool cu_between_5_and_6 = cu_new_h >= 5.0f && cu_new_h < 6.0f;
  tuple.cu_r = cu_chroma * static_cast<float>(cu_between_0_and_1 || cu_between_5_and_6) +
               cu_x * static_cast<float>(cu_between_1_and_2 || cu_between_4_and_5) + cu_new_m;
  tuple.cu_g = cu_chroma * static_cast<float>(cu_between_1_and_2 || cu_between_2_and_3) +
               cu_x * static_cast<float>(cu_between_0_and_1 || cu_between_3_and_4) + cu_new_m;
  tuple.cu_b = cu_chroma * static_cast<float>(cu_between_3_and_4 || cu_between_4_and_5) +
               cu_x * static_cast<float>(cu_between_2_and_3 || cu_between_5_and_6) + cu_new_m;
  return tuple;
}

template <typename T>
__global__ void CalAdjustHueKernel(const size_t cu_input_elements, const T *cu_input, T *cu_output,
                                   const float *cu_hue_delta) {
  for (int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3; idx < cu_input_elements;
       idx += gridDim.x * blockDim.x * 3) {
    const HsvTuple hsv = rgb2hsv_cuda(static_cast<float>(cu_input[idx]), static_cast<float>(cu_input[idx + 1]),
                                      static_cast<float>(cu_input[idx + 2]));
    float cu_new_h = hsv.cu_h;
    float cu_new_s = hsv.cu_s;
    float cu_new_v = hsv.cu_v;
    const float cu_delta = *cu_hue_delta;
    cu_new_h = fmodf(hsv.cu_h + cu_delta, 1.0f);
    if (cu_new_h < 0.0f) {
      cu_new_h = fmodf(1.0f + cu_new_h, 1.0f);
    }
    const RgbTuple rgb = hsv2rgb_cuda(cu_new_h, cu_new_s, cu_new_v);
    cu_output[idx] = static_cast<T>(rgb.cu_r);
    cu_output[idx + 1] = static_cast<T>(rgb.cu_g);
    cu_output[idx + 2] = static_cast<T>(rgb.cu_b);
  }
}

template <typename T>
cudaError_t CalAdjusthue(const int input_elements, const T *input, T *output, const float *hue_delta,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalAdjustHueKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_elements, input, output, hue_delta);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalAdjusthue<float>(const int input_elements, const float *input, float *output,
                                                         const float *hue_delta, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalAdjusthue<half>(const int input_elements, const half *input, half *output,
                                                        const float *hue_delta, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalAdjusthue<double>(const int input_elements, const double *input, double *output,
                                                          const float *hue_delta, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
