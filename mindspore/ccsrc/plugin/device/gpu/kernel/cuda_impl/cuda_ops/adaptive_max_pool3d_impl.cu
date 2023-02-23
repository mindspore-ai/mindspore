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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool3d_impl.cuh"
#include "include/cuda_fp16.h"

__device__ inline int64_t start_index(int64_t a, int64_t b, int64_t c) { return floorf(static_cast<float>(a * c) / b); }

__device__ inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return ceilf(static_cast<float>((a + 1) * c) / b);
}

template <typename T>
__device__ __forceinline__ bool IsNan(const T val) {
  return isnan(val);
}

template <>
__device__ __forceinline__ bool IsNan(const half val) {
  return __hisnan(val);
}

template <typename T>
__global__ void AdaptiveMaxPool3DKernel(const int64_t size, const int64_t channels, const int64_t input_depth,
                                        const int64_t input_height, const int64_t input_width,
                                        const int32_t *output_size, const T *input_data, T *output_data,
                                        int32_t *mask_data) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
    int64_t output_depth = static_cast<int64_t>(output_size[0]);
    int64_t output_height = static_cast<int64_t>(output_size[1]);
    int64_t output_width = static_cast<int64_t>(output_size[2]);

    int64_t pw = pos % output_width;
    int64_t ph = (pos / output_width) % output_height;
    int64_t pd = (pos / output_width / output_height) % output_depth;
    int64_t c = (pos / output_width / output_height / output_depth) % channels;
    int64_t batch_idx = pos / output_width / output_height / output_depth / channels;

    int64_t dstart = start_index(pd, output_depth, input_depth);
    int64_t dend = end_index(pd, output_depth, input_depth);

    int64_t hstart = start_index(ph, output_height, input_height);
    int64_t hend = end_index(ph, output_height, input_height);

    int64_t wstart = start_index(pw, output_width, input_width);
    int64_t wend = end_index(pw, output_width, input_width);

    const T *input_ptr = input_data + (batch_idx * channels + c) * input_depth * input_height * input_width;
    int64_t max_index = (dstart * input_height + hstart) * input_width + wstart;
    T max_data = input_ptr[max_index];

    for (int64_t d = dstart; d < dend; d++) {
      for (int64_t h = hstart; h < hend; h++) {
        for (int64_t w = wstart; w < wend; w++) {
          int64_t index = (d * input_height + h) * input_width + w;
          if (max_data < input_ptr[index] || IsNan(input_ptr[index])) {
            max_index = index;
            max_data = input_ptr[index];
          }
        }
      }
    }
    output_data[pos] = max_data;
    mask_data[pos] = static_cast<int32_t>(max_index);
  }
}

template <typename T>
cudaError_t ApplyAdaptiveMaxPool3D(const int64_t size, const int64_t channels, const int64_t input_depth,
                                   const int64_t input_height, const int64_t input_width, const T *input_data,
                                   const int32_t *output_size, T *output_data, int32_t *mask_data,
                                   const uint32_t device_id, cudaStream_t cuda_stream) {
  AdaptiveMaxPool3DKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, channels, input_depth, input_height, input_width, output_size, input_data, output_data, mask_data);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<half>(const int64_t size, const int64_t channels,
                                                                  const int64_t input_depth, const int64_t input_height,
                                                                  const int64_t input_width, const half *input_data,
                                                                  const int32_t *output_size, half *output_data,
                                                                  int32_t *mask_data, const uint32_t device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<float>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const float *input_data, const int32_t *output_size, float *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<double>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const double *input_data, const int32_t *output_size, double *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<int8_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const int8_t *input_data, const int32_t *output_size, int8_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<int16_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const int16_t *input_data, const int32_t *output_size, int16_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<int32_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const int32_t *input_data, const int32_t *output_size, int32_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<int64_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const int64_t *input_data, const int32_t *output_size, int64_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<uint8_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const uint8_t *input_data, const int32_t *output_size, uint8_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<uint16_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const uint16_t *input_data, const int32_t *output_size, uint16_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<uint32_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const uint32_t *input_data, const int32_t *output_size, uint32_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool3D<uint64_t>(
  const int64_t size, const int64_t channels, const int64_t input_depth, const int64_t input_height,
  const int64_t input_width, const uint64_t *input_data, const int32_t *output_size, uint64_t *output_data,
  int32_t *mask_data, const uint32_t device_id, cudaStream_t cuda_stream);
