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
#include "resize_area_impl.cuh"
#include "include/cuda_fp16.h"

__device__ int32_t BoundResizeArea(int32_t val, int32_t limit) { return min(limit - 1, max(int32_t{0}, val)); }

__global__ void ComputeInterpolation(ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
                                     const int32_t in_height, const int32_t in_width, const int32_t out_height,
                                     const int32_t out_width, float height_scale, float width_scale,
                                     bool align_corners) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_height + out_width;
       pos += gridDim.x * blockDim.x) {
    const int32_t pair_in_length[2] = {in_height, in_width};
    ResizeAreaCachedInterpolation *pair_interps[2] = {y_interps, x_interps};
    float pair_scale[2] = {height_scale, width_scale};
    int32_t in_length;
    ResizeAreaCachedInterpolation *interps;
    size_t offset;
    float scale;
    size_t mode = min(size_t(1), pos / out_height);
    in_length = pair_in_length[mode];
    interps = pair_interps[mode];
    scale = pair_scale[mode];
    offset = mode > 0 ? pos - out_height : pos;
    float transit_0 = offset * scale;
    float transit_1 = (offset + 1) * scale;
    size_t v = floor(transit_0);
    interps[offset].start = v;
    interps[offset].start_scale =
      v < transit_0 ? (v + 1 > transit_1 ? scale : v + 1 - transit_0) : (v + 1 > transit_1 ? transit_1 - v : 1.0);
    v = ceil(transit_1);
    interps[offset].end = v;
    v = interps[offset].end - 1;
    interps[offset].end_minus_one_scale =
      v < transit_0 ? (v + 1 > transit_1 ? scale : v + 1 - transit_0) : (v + 1 > transit_1 ? transit_1 - v : 1.0);
    interps[offset].needs_bounding = BoundResizeArea(interps[offset].start, in_length) != interps[offset].start ||
                                     BoundResizeArea(interps[offset].end, in_length) != (interps[offset].end - 1);
  }
  return;
}

template <typename T>
__global__ void PatchSum(const T *images, const ResizeAreaCachedInterpolation *x_interps,
                         const ResizeAreaCachedInterpolation *y_interps, float *output, int32_t batch_size,
                         const int32_t channels, const int32_t out_height, const int32_t out_width, const float scale,
                         const int32_t in_height, const int32_t in_width) {
#define BOUND_IF_NEEDED(x, y, NeedsBounding) (NeedsBounding ? BoundResizeArea(x, y) : (x))
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < batch_size * channels * out_height * out_width;
       pos += gridDim.x * blockDim.x) {
    size_t tem_pos = pos;
    size_t image_id = tem_pos / (channels * out_height * out_width);
    tem_pos = tem_pos % (channels * out_height * out_width);
    size_t y_id = tem_pos / (out_width * channels);
    tem_pos = tem_pos % (out_width * channels);
    size_t x_id = tem_pos / channels;
    size_t channel_id = tem_pos % channels;
    ResizeAreaCachedInterpolation x_interp = x_interps[x_id];
    ResizeAreaCachedInterpolation y_interp = y_interps[y_id];
    size_t start_offset, y_offset;
    start_offset = image_id * in_height * in_width * channels;
    float sum = 0;

    y_offset = start_offset + BOUND_IF_NEEDED(y_interp.start, in_height, y_interp.needs_bounding) * in_width * channels;
    float scale_x = x_interp.start_scale;
    float sum_y =
      static_cast<float>(
        images[y_offset + channels * BOUND_IF_NEEDED(x_interp.start, in_width, x_interp.needs_bounding) + channel_id]) *
      scale_x;
    if (x_interp.start + 1 != x_interp.end) {
      for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
        sum_y += static_cast<float>(
          images[y_offset + channels * BOUND_IF_NEEDED(x, in_width, x_interp.needs_bounding) + channel_id]);
      }
      scale_x = x_interp.end_minus_one_scale;
      sum_y += static_cast<float>(
                 images[y_offset + channels * BOUND_IF_NEEDED(x_interp.end - 1, in_width, x_interp.needs_bounding) +
                        channel_id]) *
               scale_x;
    }
    sum += sum_y * y_interp.start_scale;
    if (y_interp.start + 1 != y_interp.end) {
      for (size_t y = y_interp.start + 1; y < y_interp.end - 1; ++y) {
        y_offset = start_offset + BOUND_IF_NEEDED(y, in_height, y_interp.needs_bounding) * in_width * channels;
        scale_x = x_interp.start_scale;
        sum_y = static_cast<float>(
                  images[y_offset + channels * BOUND_IF_NEEDED(x_interp.start, in_width, x_interp.needs_bounding) +
                         channel_id]) *
                scale_x;
        if (x_interp.start + 1 != x_interp.end) {
          for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
            sum_y += static_cast<float>(
              images[y_offset + channels * BOUND_IF_NEEDED(x, in_width, x_interp.needs_bounding) + channel_id]);
          }
          scale_x = x_interp.end_minus_one_scale;
          sum_y += static_cast<float>(
                     images[y_offset + channels * BOUND_IF_NEEDED(x_interp.end - 1, in_width, x_interp.needs_bounding) +
                            channel_id]) *
                   scale_x;
        }
        sum += sum_y;
      }
      y_offset =
        start_offset + BOUND_IF_NEEDED(y_interp.end - 1, in_height, y_interp.needs_bounding) * in_width * channels;
      scale_x = x_interp.start_scale;
      sum_y = static_cast<float>(
                images[y_offset + channels * BOUND_IF_NEEDED(x_interp.start, in_width, x_interp.needs_bounding) +
                       channel_id]) *
              scale_x;
      if (x_interp.start + 1 != x_interp.end) {
        for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
          sum_y += static_cast<float>(
            images[y_offset + channels * BOUND_IF_NEEDED(x, in_width, x_interp.needs_bounding) + channel_id]);
        }
        scale_x = x_interp.end_minus_one_scale;
        sum_y += static_cast<float>(
                   images[y_offset + channels * BOUND_IF_NEEDED(x_interp.end - 1, in_width, x_interp.needs_bounding) +
                          channel_id]) *
                 scale_x;
      }
      sum += sum_y * y_interp.end_minus_one_scale;
    }
    output[pos] = sum * scale;
  }
  return;
}

// half
template <>
__global__ void PatchSum(const half *images, const ResizeAreaCachedInterpolation *x_interps,
                         const ResizeAreaCachedInterpolation *y_interps, float *output, int32_t batch_size,
                         const int32_t channels, const int32_t out_height, const int32_t out_width, const float scale,
                         const int32_t in_height, const int32_t in_width) {
#define BOUND_IF_NEEDED(x, y, NeedsBounding) (NeedsBounding ? BoundResizeArea(x, y) : (x))
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < batch_size * channels * out_height * out_width;
       pos += gridDim.x * blockDim.x) {
    size_t tem_pos = pos;
    size_t image_id = tem_pos / (channels * out_height * out_width);
    tem_pos = tem_pos % (channels * out_height * out_width);
    size_t y_id = tem_pos / (out_width * channels);
    tem_pos = tem_pos % (out_width * channels);
    size_t x_id = tem_pos / channels;
    size_t channel_id = tem_pos % channels;
    ResizeAreaCachedInterpolation x_interp = x_interps[x_id];
    ResizeAreaCachedInterpolation y_interp = y_interps[y_id];
    size_t start_offset, y_offset;
    start_offset = image_id * in_height * in_width * channels;
    float sum = 0;
    y_offset = start_offset + BOUND_IF_NEEDED(y_interp.start, in_height, y_interp.needs_bounding) * in_width * channels;
    float scale_x = x_interp.start_scale;
    float sum_y =
      __half2float(
        images[y_offset + channels * BOUND_IF_NEEDED(x_interp.start, in_width, x_interp.needs_bounding) + channel_id]) *
      scale_x;
    if (x_interp.start + 1 != x_interp.end) {
      for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
        sum_y += __half2float(
          images[y_offset + channels * BOUND_IF_NEEDED(x, in_width, x_interp.needs_bounding) + channel_id]);
      }
      scale_x = x_interp.end_minus_one_scale;
      sum_y +=
        __half2float(images[y_offset + channels * BOUND_IF_NEEDED(x_interp.end - 1, in_width, x_interp.needs_bounding) +
                            channel_id]) *
        scale_x;
    }
    sum += sum_y * y_interp.start_scale;
    if (y_interp.start + 1 != y_interp.end) {
      for (size_t y = y_interp.start + 1; y < y_interp.end - 1; ++y) {
        y_offset = start_offset + BOUND_IF_NEEDED(y, in_height, y_interp.needs_bounding) * in_width * channels;
        scale_x = x_interp.start_scale;
        sum_y =
          __half2float(images[y_offset + channels * BOUND_IF_NEEDED(x_interp.start, in_width, x_interp.needs_bounding) +
                              channel_id]) *
          scale_x;
        if (x_interp.start + 1 != x_interp.end) {
          for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
            sum_y += __half2float(
              images[y_offset + channels * BOUND_IF_NEEDED(x, in_width, x_interp.needs_bounding) + channel_id]);
          }
          scale_x = x_interp.end_minus_one_scale;
          sum_y += __half2float(
                     images[y_offset + channels * BOUND_IF_NEEDED(x_interp.end - 1, in_width, x_interp.needs_bounding) +
                            channel_id]) *
                   scale_x;
        }
        sum += sum_y;
      }
      y_offset =
        start_offset + BOUND_IF_NEEDED(y_interp.end - 1, in_height, y_interp.needs_bounding) * in_width * channels;
      scale_x = x_interp.start_scale;
      sum_y =
        __half2float(images[y_offset + channels * BOUND_IF_NEEDED(x_interp.start, in_width, x_interp.needs_bounding) +
                            channel_id]) *
        scale_x;
      if (x_interp.start + 1 != x_interp.end) {
        for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
          sum_y += __half2float(
            images[y_offset + channels * BOUND_IF_NEEDED(x, in_width, x_interp.needs_bounding) + channel_id]);
        }
        scale_x = x_interp.end_minus_one_scale;
        sum_y += __half2float(
                   images[y_offset + channels * BOUND_IF_NEEDED(x_interp.end - 1, in_width, x_interp.needs_bounding) +
                          channel_id]) *
                 scale_x;
      }
      sum += sum_y * y_interp.end_minus_one_scale;
    }
    output[pos] = sum * scale;
  }
  return;
}

float Scaling(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

template <typename T>
cudaError_t CalResizeArea(const T *images, ResizeAreaCachedInterpolation *x_interps,
                          ResizeAreaCachedInterpolation *y_interps, float *output, int32_t batch_size,
                          const int32_t channels, const int32_t out_height, const int32_t out_width,
                          const int32_t in_height, const int32_t in_width, bool align_corners,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  float height_scale = Scaling(in_height, out_height, align_corners);
  float width_scale = Scaling(in_width, out_width, align_corners);
  float scale = 1.0 / (height_scale * width_scale);
  ComputeInterpolation<<<CUDA_BLOCKS(device_id, out_height + out_width), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    x_interps, y_interps, in_height, in_width, out_height, out_width, height_scale, width_scale, align_corners);
  PatchSum<<<CUDA_BLOCKS(device_id, batch_size * out_height * out_width * channels), CUDA_THREADS(device_id), 0,
             cuda_stream>>>(images, x_interps, y_interps, output, batch_size, channels, out_height, out_width, scale,
                            in_height, in_width);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalResizeArea<int8_t>(
  const int8_t *images, ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
  float *output, int32_t batch_size, const int32_t channels, const int32_t out_height, const int32_t out_width,
  const int32_t in_height, const int32_t in_width, bool align_corners, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<uint8_t>(
  const uint8_t *images, ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
  float *output, int32_t batch_size, const int32_t channels, const int32_t out_height, const int32_t out_width,
  const int32_t in_height, const int32_t in_width, bool align_corners, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<int16_t>(
  const int16_t *images, ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
  float *output, int32_t batch_size, const int32_t channels, const int32_t out_height, const int32_t out_width,
  const int32_t in_height, const int32_t in_width, bool align_corners, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<uint16_t>(
  const uint16_t *images, ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
  float *output, int32_t batch_size, const int32_t channels, const int32_t out_height, const int32_t out_width,
  const int32_t in_height, const int32_t in_width, bool align_corners, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<int32_t>(
  const int32_t *images, ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
  float *output, int32_t batch_size, const int32_t channels, const int32_t out_height, const int32_t out_width,
  const int32_t in_height, const int32_t in_width, bool align_corners, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<int64_t>(
  const int64_t *images, ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
  float *output, int32_t batch_size, const int32_t channels, const int32_t out_height, const int32_t out_width,
  const int32_t in_height, const int32_t in_width, bool align_corners, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<half>(const half *images, ResizeAreaCachedInterpolation *x_interps,
                                                         ResizeAreaCachedInterpolation *y_interps, float *output,
                                                         int32_t batch_size, const int32_t channels,
                                                         const int32_t out_height, const int32_t out_width,
                                                         const int32_t in_height, const int32_t in_width,
                                                         bool align_corners, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<float>(const float *images, ResizeAreaCachedInterpolation *x_interps,
                                                          ResizeAreaCachedInterpolation *y_interps, float *output,
                                                          int32_t batch_size, const int32_t channels,
                                                          const int32_t out_height, const int32_t out_width,
                                                          const int32_t in_height, const int32_t in_width,
                                                          bool align_corners, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeArea<double>(
  const double *images, ResizeAreaCachedInterpolation *x_interps, ResizeAreaCachedInterpolation *y_interps,
  float *output, int32_t batch_size, const int32_t channels, const int32_t out_height, const int32_t out_width,
  const int32_t in_height, const int32_t in_width, bool align_corners, const uint32_t &device_id,
  cudaStream_t cuda_stream);
