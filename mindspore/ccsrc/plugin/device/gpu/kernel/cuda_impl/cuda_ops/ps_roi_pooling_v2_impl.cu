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

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ps_roi_pooling_v2_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void PSROIPoolForwardV2(const int nthreads, const T *input, const T spatial_scale, const int feature_height,
                                   const int feature_width, const int feature_channels, const int pooled_height,
                                   const int pooled_width, const int group_size, const int output_channels,
                                   const T *roi_boxes, T *output_data) {
  const int elements_per_roi_box = 5;
  constexpr float zero = 0;
  // Loop over the outputs of forward operator.
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    int width_offset_n = index % pooled_width;
    int height_offset_n = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_channels;
    int n = index / pooled_width / pooled_height / output_channels;

    const T *offset_rois = roi_boxes + n * elements_per_roi_box;
    int roi_batch_ind = static_cast<int>(offset_rois[0]);
    // floor round not support half
    T roi_start_width = static_cast<T>(round(static_cast<float>(offset_rois[1] * spatial_scale)));
    T roi_start_height = static_cast<T>(round(static_cast<float>(offset_rois[2] * spatial_scale)));
    T roi_end_width = static_cast<T>(round(static_cast<float>(offset_rois[3] * spatial_scale)));
    T roi_end_height = static_cast<T>(round(static_cast<float>(offset_rois[4] * spatial_scale)));

    // Force malformed ROIs to be 1x1
    T roi_width = max(static_cast<float>(roi_end_width - roi_start_width), 0.1);  // avoid 0
    T roi_height = max(static_cast<float>(roi_end_height - roi_start_height), 0.1);

    T bin_height = (T)(roi_height) / (T)(pooled_height);
    T bin_width = (T)(roi_width) / (T)(pooled_width);

    int pooling_start_x = static_cast<int>(floor(static_cast<float>(static_cast<T>(height_offset_n) * bin_height)));
    int pooling_start_y = static_cast<int>(floor(static_cast<float>(static_cast<T>(width_offset_n) * bin_width)));
    int pooling_end_x = static_cast<int>(ceil(static_cast<float>(static_cast<T>(height_offset_n + 1) * bin_height)));
    int pooling_end_y = static_cast<int>(ceil(static_cast<float>(static_cast<T>(width_offset_n + 1) * bin_width)));

    // Add roi offsets and clip to input boundaries
    pooling_start_x = min(max(pooling_start_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_end_x = min(max(pooling_end_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_start_y = min(max(pooling_start_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    pooling_end_y = min(max(pooling_end_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

    int gw = width_offset_n;
    int gh = height_offset_n;
    int c = (ctop * group_size + gh) * group_size + gw;

    const T *offset_input = input + (roi_batch_ind * feature_channels + c) * feature_height * feature_width;
    T out_sum = T(zero);
    for (int h = pooling_start_x; h < pooling_end_x; ++h) {
      for (int w = pooling_start_y; w < pooling_end_y; ++w) {
        int bottom_index = h * feature_width + w;
        out_sum += offset_input[bottom_index];
      }
    }
    T bin_area = static_cast<T>((pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y));
    output_data[index] = is_empty ? T(zero) : out_sum / bin_area;
    }
}
template <>
__global__ void PSROIPoolForwardV2(const int nthreads, const double *input, const double spatial_scale,
                                   const int feature_height, const int feature_width, const int feature_channels,
                                   const int pooled_height, const int pooled_width, const int group_size,
                                   const int output_channels, const double *roi_boxes, double *output_data) {
  const int elements_per_roi_box = 5;
  constexpr float zero = 0;
  // Loop over the outputs of forward operator.
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    int width_offset_n = index % pooled_width;
    int height_offset_n = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_channels;
    int n = index / pooled_width / pooled_height / output_channels;

    const double *offset_rois = roi_boxes + n * elements_per_roi_box;
    int roi_batch_ind = static_cast<int>(offset_rois[0]);
    // floor round not support half
    double roi_start_width = static_cast<double>(round(static_cast<double>(offset_rois[1] * spatial_scale)));
    double roi_start_height = static_cast<double>(round(static_cast<double>(offset_rois[2] * spatial_scale)));
    double roi_end_width = static_cast<double>(round(static_cast<double>(offset_rois[3] * spatial_scale)));
    double roi_end_height = static_cast<double>(round(static_cast<double>(offset_rois[4] * spatial_scale)));

    // Force malformed ROIs to be 1x1
    double roi_width = max(static_cast<double>(roi_end_width - roi_start_width), 0.1);  // avoid 0
    double roi_height = max(static_cast<double>(roi_end_height - roi_start_height), 0.1);

    double bin_height = roi_height / static_cast<double>(pooled_height);
    double bin_width = roi_width / static_cast<double>(pooled_width);

    int pooling_start_x =
                       static_cast<int>(floor(static_cast<double>(static_cast<double>(height_offset_n) * bin_height)));
    int pooling_start_y = static_cast<int>(floor(static_cast<double>(static_cast<double>(width_offset_n) * bin_width)));
    int pooling_end_x =
                     static_cast<int>(ceil(static_cast<double>(static_cast<double>(height_offset_n + 1) * bin_height)));
    int pooling_end_y =
                     static_cast<int>(ceil(static_cast<double>(static_cast<double>(width_offset_n + 1) * bin_width)));

    // Add roi offsets and clip to input boundaries
    pooling_start_x = min(max(pooling_start_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_end_x = min(max(pooling_end_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_start_y = min(max(pooling_start_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    pooling_end_y = min(max(pooling_end_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

    int gw = width_offset_n;
    int gh = height_offset_n;
    int c = (ctop * group_size + gh) * group_size + gw;

    const double *offset_input = input + (roi_batch_ind * feature_channels + c) * feature_height * feature_width;
    double out_sum = static_cast<double>(zero);
    for (int h = pooling_start_x; h < pooling_end_x; ++h) {
      for (int w = pooling_start_y; w < pooling_end_y; ++w) {
        int bottom_index = h * feature_width + w;
        out_sum += offset_input[bottom_index];
      }
    }
    double bin_area = static_cast<double>((pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y));
    output_data[index] = is_empty ? static_cast<double>(zero) : out_sum / bin_area;
    }
}

template <>
__global__ void PSROIPoolForwardV2(const int nthreads, const float *input, const float spatial_scale,
                                   const int feature_height, const int feature_width, const int feature_channels,
                                   const int pooled_height, const int pooled_width, const int group_size,
                                   const int output_channels, const float *roi_boxes, float *output_data) {
  const int elements_per_roi_box = 5;
  constexpr float zero = 0;
  // Loop over the outputs of forward operator.
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    int width_offset_n = index % pooled_width;
    int height_offset_n = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_channels;
    int n = index / pooled_width / pooled_height / output_channels;

    const float *offset_rois = roi_boxes + n * elements_per_roi_box;
    int roi_batch_ind = static_cast<int>(offset_rois[0]);
    // floor round not support half
    float roi_start_width = static_cast<float>(round(static_cast<float>(offset_rois[1] * spatial_scale)));
    float roi_start_height = static_cast<float>(round(static_cast<float>(offset_rois[2] * spatial_scale)));
    float roi_end_width = static_cast<float>(round(static_cast<float>(offset_rois[3] * spatial_scale)));
    float roi_end_height = static_cast<float>(round(static_cast<float>(offset_rois[4] * spatial_scale)));

    // Force malformed ROIs to be 1x1
    float roi_width = max(static_cast<float>(roi_end_width - roi_start_width), 0.1);  // avoid 0
    float roi_height = max(static_cast<float>(roi_end_height - roi_start_height), 0.1);

    float bin_height = roi_height / static_cast<float>(pooled_height);
    float bin_width = roi_width / static_cast<float>(pooled_width);

    int pooling_start_x = static_cast<int>(floor(static_cast<float>(static_cast<float>(height_offset_n) * bin_height)));
    int pooling_start_y = static_cast<int>(floor(static_cast<float>(static_cast<float>(width_offset_n) * bin_width)));
    int pooling_end_x =
                     static_cast<int>(ceil(static_cast<float>(static_cast<float>(height_offset_n + 1) * bin_height)));
    int pooling_end_y = static_cast<int>(ceil(static_cast<float>(static_cast<float>(width_offset_n + 1) * bin_width)));

    // Add roi offsets and clip to input boundaries
    pooling_start_x = min(max(pooling_start_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_end_x = min(max(pooling_end_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_start_y = min(max(pooling_start_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    pooling_end_y = min(max(pooling_end_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

    int gw = width_offset_n;
    int gh = height_offset_n;
    int c = (ctop * group_size + gh) * group_size + gw;

    const float *offset_input = input + (roi_batch_ind * feature_channels + c) * feature_height * feature_width;
    float out_sum = static_cast<float>(zero);
    for (int h = pooling_start_x; h < pooling_end_x; ++h) {
      for (int w = pooling_start_y; w < pooling_end_y; ++w) {
        int bottom_index = h * feature_width + w;
        out_sum += offset_input[bottom_index];
      }
    }
    float bin_area = static_cast<float>((pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y));
    output_data[index] = is_empty ? static_cast<float>(zero) : out_sum / bin_area;
    }
}

template <>
__global__ void PSROIPoolForwardV2(const int nthreads, const half *input, const half spatial_scale,
                                   const int feature_height, const int feature_width, const int feature_channels,
                                   const int pooled_height, const int pooled_width, const int group_size,
                                   const int output_channels,  const half *roi_boxes, half *output_data) {
  const int elements_per_roi_box = 5;
  constexpr float zero = 0;
  // Loop over the outputs of forward operator.
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    int width_offset_n = index % pooled_width;
    int height_offset_n = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_channels;
    int n = index / pooled_width / pooled_height / output_channels;

    const half *offset_rois = roi_boxes + n * elements_per_roi_box;
    int roi_batch_ind = static_cast<int>(floor(static_cast<float>(offset_rois[0])));
    // floor round not support half
    half roi_start_width = static_cast<half>(round(static_cast<float>(offset_rois[1] * spatial_scale)));
    half roi_start_height = static_cast<half>(round(static_cast<float>(offset_rois[2] * spatial_scale)));
    half roi_end_width = static_cast<half>(round(static_cast<float>(offset_rois[3] * spatial_scale)));
    half roi_end_height = static_cast<half>(round(static_cast<float>(offset_rois[4] * spatial_scale)));

    // Force malformed ROIs to be 1x1
    half roi_width = max(static_cast<half>(roi_end_width - roi_start_width), 0.1);  // avoid 0
    half roi_height = max(static_cast<half>(roi_end_height - roi_start_height), 0.1);

    half bin_height = roi_height / static_cast<half>(pooled_height);
    half bin_width = roi_width / static_cast<half>(pooled_width);

    int pooling_start_x = static_cast<int>(floor(static_cast<float>(static_cast<half>(height_offset_n) * bin_height)));
    int pooling_start_y = static_cast<int>(floor(static_cast<float>(static_cast<half>(width_offset_n) * bin_width)));
    int pooling_end_x = static_cast<int>(ceil(static_cast<float>(static_cast<half>(height_offset_n + 1) * bin_height)));
    int pooling_end_y = static_cast<int>(ceil(static_cast<float>(static_cast<half>(width_offset_n + 1) * bin_width)));

    // Add roi offsets and clip to input boundaries
    pooling_start_x = min(max(pooling_start_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_end_x = min(max(pooling_end_x + static_cast<int>(roi_start_height), 0), feature_height - 1);
    pooling_start_y = min(max(pooling_start_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    pooling_end_y = min(max(pooling_end_y + static_cast<int>(roi_start_width), 0), feature_width - 1);
    bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

    int gw = width_offset_n;
    int gh = height_offset_n;
    int c = (ctop * group_size + gh) * group_size + gw;

    const half *offset_input = input + (roi_batch_ind * feature_channels + c) * feature_height * feature_width;
    float out_sum = static_cast<float>(zero);
    for (int h = pooling_start_x; h < pooling_end_x; ++h) {
      for (int w = pooling_start_y; w < pooling_end_y; ++w) {
        int bottom_index = h * feature_width + w;
        out_sum += static_cast<float>(offset_input[bottom_index]);
      }
    }
    float bin_area = static_cast<float>((pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y));
    output_data[index] = is_empty ? half(zero) : static_cast<half>(out_sum / bin_area);
    }
}

template <typename T>
void PSROIPoolForwardV2Launcher(const T *input, const T spatial_scale, const int output_n, const int feature_height,
                                const int feature_width, const int feature_channels, const int pooled_height,
                                const int pooled_width, const T *roi_boxes, const int group_size,
                                const int output_channels, T *output_data, cudaStream_t stream) {
  const int kThreadsPerBlock_ = 1024;
  const int output_size = output_channels * pooled_height * pooled_width * output_n;
  cudaError_t err;

  PSROIPoolForwardV2<<<(output_size + kThreadsPerBlock_ - 1) / kThreadsPerBlock_, kThreadsPerBlock_, 0, stream>>>(
  output_size, input, spatial_scale, feature_height, feature_width, feature_channels, pooled_height, pooled_width,
  group_size, output_channels, roi_boxes, output_data);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

template CUDA_LIB_EXPORT void PSROIPoolForwardV2Launcher<double>(const double *input, const double spatial_scale,
                                                                const int output_n, const int feature_height,
                                                                const int feature_width, const int feature_channels,
                                                                const int pooled_height, const int pooled_width,
                                                                const double *roi_boxes, const int group_size,
                                                                const int output_channels, double *output_data,
                                                                cudaStream_t stream);

template CUDA_LIB_EXPORT void PSROIPoolForwardV2Launcher<float>(const float *input, const float spatial_scale,
                                                                const int output_n, const int feature_height,
                                                                const int feature_width, const int feature_channels,
                                                                const int pooled_height, const int pooled_width,
                                                                const float *roi_boxes, const int group_size,
                                                                const int output_channels, float *output_data,
                                                                cudaStream_t stream);

template CUDA_LIB_EXPORT void PSROIPoolForwardV2Launcher<half>(const half *input, const half spatial_scale,
                                                               const int output_n, const int feature_height,
                                                               const int feature_width, const int feature_channels,
                                                               const int pooled_height, const int pooled_width,
                                                               const half *roi_boxes, const int group_size,
                                                               const int output_channels, half *output_data,
                                                               cudaStream_t stream);
