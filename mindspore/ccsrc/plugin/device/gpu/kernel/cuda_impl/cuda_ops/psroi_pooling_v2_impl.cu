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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/psroi_pooling_v2_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void PSROIPoolInitKernel(size_t size_init, T *input) {
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < size_init;
       thread_idx += blockDim.x * gridDim.x) {
    input[thread_idx] = static_cast<T>(.0);
  }
}

template <typename T>
__global__ void PSROIPoolBackwardV2(const int nthreads, T *input_diff, const T spatial_scale, const int feature_height,
                                    const int feature_width, const int feature_channels, const int pooled_height,
                                    const int pooled_width, const int output_channels, T *output_diff, T *roi_boxes,
                                    int batch_size, int rois_num, int group_size) {
  const int elements_per_roi_box = 5;
  // Loop over the outputs of forward operator.
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    int width_offset_n = index % pooled_width;
    int height_offset_n = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / output_channels;
    int n_batch = n / rois_num;
    int n_rois_num = n % rois_num;

    // find pooling box index
    T *p_roi_batch_index = roi_boxes + n_batch * (rois_num * elements_per_roi_box) + n_rois_num;
    int roi_batch_index = static_cast<int>(*p_roi_batch_index);

    T *p_roi_start_width = p_roi_batch_index + rois_num;
    T roi_start_width = static_cast<T>(roundf((*p_roi_start_width) * spatial_scale));

    T *p_roi_start_height = p_roi_start_width + rois_num;
    T roi_start_height = static_cast<T>(roundf((*p_roi_start_height) * spatial_scale));

    T *p_roi_end_width = p_roi_start_height + rois_num;
    T roi_end_width = static_cast<T>(roundf((*p_roi_end_width) * spatial_scale));

    T *p_roi_end_height = p_roi_end_width + rois_num;
    T roi_end_height = static_cast<T>(roundf((*p_roi_end_height) * spatial_scale));

    // let min roi len and width bigger than 0.1
    T roi_width = max(roi_end_width - roi_start_width, 0.1);
    T roi_height = max(roi_end_height - roi_start_height, 0.1);

    // Compute bin_width and bin_height
    T bin_height = roi_height / static_cast<T>(pooled_height);
    T bin_width = roi_width / static_cast<T>(pooled_width);
    // compute pooling area's position
    int pooling_start_x = floor(static_cast<float>(static_cast<T>(height_offset_n) * bin_height + roi_start_height));
    int pooling_start_y = floor(static_cast<float>(static_cast<T>(width_offset_n) * bin_width + roi_start_width));
    int pooling_end_x = ceil(static_cast<float>(static_cast<T>(height_offset_n + 1) * bin_height + roi_start_height));
    int pooling_end_y = ceil(static_cast<float>(static_cast<T>(width_offset_n + 1) * bin_width + roi_start_width));
    // Add roi offsets and clip to input boundaries
    pooling_start_x = min(max(pooling_start_x, 0), feature_height);
    pooling_end_x = min(max(pooling_end_x, 0), feature_height);
    pooling_start_y = min(max(pooling_start_y, 0), feature_width);
    pooling_end_y = min(max(pooling_end_y, 0), feature_width);
    bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

    int c = index % (pooled_height * pooled_width * output_channels);

    T *offset_bottom_diff = output_diff + (roi_batch_index * feature_channels + c) * feature_height * feature_width;
    T bin_area = (pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y);
    T diff_val = is_empty ? T(0.) : input_diff[index] / bin_area;
    for (int h = pooling_start_x; h < pooling_end_x; ++h) {
      for (int w = pooling_start_y; w < pooling_end_y; ++w) {
        int bottom_index = h * feature_width + w;
        MsAtomicAdd(offset_bottom_diff + bottom_index, diff_val);
      }
    }
  }
}

template <typename T>
void PSROIPoolBackwardV2Launcher(T *input_diff, const int batch_size, const int output_n, const T spatial_scale,
                                 const int feature_channels, const int feature_height, const int feature_width,
                                 const int pooled_width, const int pooled_height, const int output_channels,
                                 T *output_diff, T *roi_boxes, cudaStream_t stream, int rois_num, int group_size) {
  size_t size_init = batch_size * feature_channels * feature_height * feature_width;
  PSROIPoolInitKernel<<<GET_BLOCKS(size_init), GET_THREADS, 0, stream>>>(size_init, output_diff);

  const int kThreadsPerBlock_ = 1024;
  const int output_size = output_channels * pooled_height * pooled_width * output_n;
  cudaError_t err;

  PSROIPoolBackwardV2<<<(output_size + kThreadsPerBlock_ - 1) / kThreadsPerBlock_, kThreadsPerBlock_, 0, stream>>>(
    output_size, input_diff, spatial_scale, feature_height, feature_width, feature_channels, pooled_height,
    pooled_width, output_channels, output_diff, roi_boxes, batch_size, rois_num, group_size);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

template CUDA_LIB_EXPORT void PSROIPoolBackwardV2Launcher<float>(
  float *input_diff, const int batch_size, const int output_n, const float spatial_scale, const int feature_channels,
  const int feature_height, const int feature_width, const int pooled_width, const int pooled_height,
  const int output_channels, float *output_diff, float *roi_boxes, cudaStream_t stream, int rois_num, int group_size);

template CUDA_LIB_EXPORT void PSROIPoolBackwardV2Launcher<half>(
  half *input_diff, const int batch_size, const int output_n, const half spatial_scale, const int feature_channels,
  const int feature_height, const int feature_width, const int pooled_width, const int pooled_height,
  const int output_channels, half *output_diff, half *roi_boxes, cudaStream_t stream, int rois_num, int group_size);
