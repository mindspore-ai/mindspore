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

#include "adjustsaturation_impl.cuh"
#include <algorithm>
#include <cmath>

template <typename T>
__device__ __forceinline__ void rgb2hsv_cuda(const T cu_r, const T cu_g, const T cu_b, T *cu_h, T *cu_s, T *cu_v) {
  *cu_v = max(cu_r, max(cu_g, cu_b));
  const T cu_m = min(cu_r, min(cu_g, cu_b));
  const T cu_chroma = (*cu_v) - cu_m;
  if (cu_chroma > 0.0) {
    if ((*cu_v) == cu_r) {
      const T cu_num = (cu_g - cu_b) / cu_chroma;
      const T cu_sign = copysign(static_cast<T>(1), cu_num);
      *cu_h =
        ((cu_sign < 0.0) * 6.0 + cu_sign * fmodf(static_cast<float>(cu_sign * cu_num), static_cast<float>(6.0))) / 6.0;
    } else if ((*cu_v) == cu_g) {
      *cu_h = ((cu_b - cu_r) / cu_chroma + 2.0) / 6.0;
    } else {
      *cu_h = ((cu_r - cu_g) / cu_chroma + 4.0) / 6.0;
    }
  } else {
    *cu_h = 0.0;
  }
  if ((*cu_v) > 0.0) {
    *cu_s = cu_chroma / (*cu_v);
  } else {
    *cu_s = 0.0;
  }
  return;
}

template <typename T>
__device__ __forceinline__ void hsv2rgb_cuda(const T cu_h, const T cu_s, const T cu_v, T *cu_r, T *cu_g, T *cu_b) {
  const T cu_new_h = cu_h * 6.0;
  const T cu_chroma = cu_v * cu_s;
  const T cu_x = cu_chroma * (1.0f - abs(fmodf(static_cast<float>(cu_new_h), static_cast<float>(2.0)) - 1.0));
  const T cu_new_m = cu_v - cu_chroma;
  const bool cu_between_0_and_1 = cu_new_h >= 0.0f && cu_new_h < 1.0f;
  const bool cu_between_1_and_2 = cu_new_h >= 1.0f && cu_new_h < 2.0f;
  const bool cu_between_2_and_3 = cu_new_h >= 2.0f && cu_new_h < 3.0f;
  const bool cu_between_3_and_4 = cu_new_h >= 3.0f && cu_new_h < 4.0f;
  const bool cu_between_4_and_5 = cu_new_h >= 4.0f && cu_new_h < 5.0f;
  const bool cu_between_5_and_6 = cu_new_h >= 5.0f && cu_new_h < 6.0f;
  *cu_r = cu_chroma * (cu_between_0_and_1 || cu_between_5_and_6) + cu_x * (cu_between_1_and_2 || cu_between_4_and_5) +
         cu_new_m;
  *cu_g = cu_chroma * (cu_between_1_and_2 || cu_between_2_and_3) + cu_x * (cu_between_0_and_1 || cu_between_3_and_4) +
         cu_new_m;
  *cu_b = cu_chroma * (cu_between_3_and_4 || cu_between_4_and_5) + cu_x * (cu_between_2_and_3 || cu_between_5_and_6) +
         cu_new_m;
  return;
}

template <typename T>
__global__ void CalAdjustSaturationKernel(const size_t tuple_elements, const int channel_num, const T *cu_input,
                                          T *cu_output, const float *cu_saturation_scale) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < tuple_elements; idx += gridDim.x * blockDim.x) {
    T cu_new_h = 0;
    T cu_new_s = 0;
    T cu_new_v = 0;
    rgb2hsv_cuda(cu_input[channel_num * idx], cu_input[channel_num * idx + 1], cu_input[channel_num * idx + 2],
                 &cu_new_h, &cu_new_s, &cu_new_v);
    const float cu_scale = *cu_saturation_scale;
    cu_new_s = min(1.0f, max(0.0f, cu_new_s * cu_scale));
    hsv2rgb_cuda(cu_new_h, cu_new_s, cu_new_v, &cu_output[channel_num * idx], &cu_output[channel_num * idx + 1],
                 &cu_output[channel_num * idx + 2]);
  }
}

template <>
__global__ void CalAdjustSaturationKernel(const size_t tuple_elements, const int channel_num, const half *cu_input,
                                          half *cu_output, const float *cu_saturation_scale) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < tuple_elements; idx += gridDim.x * blockDim.x) {
    float cu_new_h = 0;
    float cu_new_s = 0;
    float cu_new_v = 0;
    rgb2hsv_cuda(__half2float(cu_input[channel_num * idx]), __half2float(cu_input[channel_num * idx + 1]),
                 __half2float(cu_input[channel_num * idx + 2]), &cu_new_h, &cu_new_s, &cu_new_v);
    const float cu_scale = *cu_saturation_scale;
    cu_new_s = min(1.0f, max(0.0f, cu_new_s * cu_scale));
    float cu_r = 0;
    float cu_g = 0;
    float cu_b = 0;
    hsv2rgb_cuda(cu_new_h, cu_new_s, cu_new_v, &cu_r, &cu_g, &cu_b);
    cu_output[channel_num * idx] = __float2half(cu_r);
    cu_output[channel_num * idx + 1] = __float2half(cu_g);
    cu_output[channel_num * idx + 2] = __float2half(cu_b);
  }
}

template <typename T>
void CalAdjustSaturation(const int input_elements, const T *input, T *output, const float *saturation_scale,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int channel_num = 3;
  int tuple_element = input_elements / channel_num;
  int thread_num = tuple_element > 256 ? 256 : tuple_element;
  CalAdjustSaturationKernel<<<CUDA_BLOCKS_CAL(device_id, tuple_element, thread_num), thread_num, 0, cuda_stream>>>(
    tuple_element, channel_num, input, output, saturation_scale);
}

template CUDA_LIB_EXPORT void CalAdjustSaturation<float>(const int input_elements, const float *input, float *output,
                                                         const float *saturation_scale, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdjustSaturation<half>(const int input_elements, const half *input, half *output,
                                                        const float *saturation_scale, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdjustSaturation<double>(const int input_elements, const double *input, double *output,
                                                          const float *saturation_scale, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
