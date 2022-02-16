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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/psroi_pooling_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void PSROIPoolInitKernel(size_t size_init, T *input) {
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < size_init;
       thread_idx += blockDim.x * gridDim.x) {
    input[thread_idx] = static_cast<T>(.0);
  }
}

template <typename T>
__global__ void PSROIPoolForward(const int nthreads, const T* input,
    const T spatial_scale, const int feature_height, const int feature_width,
    const int feature_channels, const int pooled_height, const int pooled_width,
    const int group_size, const int output_channels,
    const T* roi_boxes, T* output_data, int* mapping_channel) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
        int width_offset_n = index % pooled_width;
        int height_offset_n = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_channels;
        int n = index / pooled_width / pooled_height / output_channels;

        roi_boxes += n * 5;
        int roi_batch_ind = roi_boxes[0];
        // floor round not support half
        T roi_start_width = static_cast<T>(round(static_cast<float>(roi_boxes[1]))) * spatial_scale;
        T roi_start_height = static_cast<T>(round(static_cast<float>(roi_boxes[2]))) * spatial_scale;
        T roi_end_width = static_cast<T>(round(static_cast<float>(roi_boxes[3]) + 1.)) * spatial_scale;
        T roi_end_height = static_cast<T>(round(static_cast<float>(roi_boxes[4]) + 1.)) * spatial_scale;

        // Force malformed ROIs to be 1x1
        T roi_width = max(roi_end_width - roi_start_width, 0.1);  // avoid 0
        T roi_height = max(roi_end_height - roi_start_height, 0.1);

        T bin_height = (T)(roi_height) / (T)(pooled_height);
        T bin_width = (T)(roi_width) / (T)(pooled_width);

        int pooling_start_x = floor(static_cast<float>(static_cast<T>(height_offset_n) *
                              bin_height + roi_start_height));
        int pooling_start_y = floor(static_cast<float>(static_cast<T>(width_offset_n) *
                              bin_width + roi_start_width));
        int pooling_end_x = ceil(static_cast<float>(static_cast<T>(height_offset_n + 1) *
                              bin_height + roi_start_height));
        int pooling_end_y = ceil(static_cast<float>(static_cast<T>(width_offset_n + 1) *
                              bin_width + roi_start_width));

        // Add roi offsets and clip to input boundaries
        pooling_start_x = min(max(pooling_start_x, 0), feature_height);
        pooling_end_x = min(max(pooling_end_x, 0), feature_height);
        pooling_start_y = min(max(pooling_start_y, 0), feature_width);
        pooling_end_y = min(max(pooling_end_y, 0), feature_width);
        bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

        int gw = width_offset_n;
        int gh = height_offset_n;
        int c = (ctop * group_size + gh) * group_size + gw;

        input += (roi_batch_ind * feature_channels + c) * feature_height * feature_width;
        T out_sum = 0;
        for (int h = pooling_start_x; h < pooling_end_x; ++h) {
            for (int w = pooling_start_y; w < pooling_end_y; ++w) {
                int bottom_index = h * feature_width + w;
                out_sum += input[bottom_index];
            }
        }
        T bin_area = (pooling_end_x - pooling_start_x) * (pooling_end_y - pooling_start_y);
        output_data[index] = is_empty ? T(0.) : out_sum / bin_area;
        mapping_channel[index] = c;
    }
}

template <typename T>
void PSROIPoolForwardLauncher(
    const T* input, const T spatial_scale, const int rois_number, const int feature_height,
    const int feature_width, const int feature_channels, const int pooled_height,
    const int pooled_width, const T* roi_boxes,
    const int group_size, const int output_channels,
    T* output_data, int* mapping_channel, cudaStream_t stream) {
    const int kThreadsPerBlock_ = 1024;
    const int output_size = output_channels * pooled_height * pooled_width * rois_number;
    cudaError_t err;

    PSROIPoolForward<<<(output_size + kThreadsPerBlock_ - 1) / kThreadsPerBlock_, kThreadsPerBlock_, 0, stream>>>(
    output_size, input, spatial_scale, feature_height, feature_width, feature_channels, pooled_height,
    pooled_width, group_size, output_channels, roi_boxes, output_data, mapping_channel);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

template CUDA_LIB_EXPORT void PSROIPoolForwardLauncher<float>(const float* input, const float spatial_scale,
                                                              const int rois_number, const int feature_height,
                                                              const int feature_width, const int feature_channels,
                                                              const int pooled_height, const int pooled_width,
                                                              const float* roi_boxes, const int group_size,
                                                              const int output_channels, float* output_data,
                                                              int* mapping_channel, cudaStream_t stream);

template CUDA_LIB_EXPORT void PSROIPoolForwardLauncher<half>(const half *input, const half spatial_scale,
                                                             const int rois_number, const int feature_height,
                                                             const int feature_width, const int feature_channels,
                                                             const int pooled_height, const int pooled_width,
                                                             const half *roi_boxes, const int group_size,
                                                             const int output_channels, half *output_data,
                                                             int* mapping_channel, cudaStream_t stream);

template <typename T>
__global__ void PSROIPoolBackward(const int nthreads, const T* input_diff,
    const int* mapping_channel, const T spatial_scale,
    const int feature_height, const int feature_width, const int feature_channels,
    const int pooled_height, const int pooled_width, const int output_channels, T* output_diff,
    const T* roi_boxes) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
        int width_offset_n = index % pooled_width;
        int height_offset_n = (index / pooled_width) % pooled_height;
        int n = index / pooled_width / pooled_height / output_channels;

        // find pooling box index
        roi_boxes += n * 5;
        int roi_batch_index = roi_boxes[0];
        T roi_start_width = static_cast<T>(round(static_cast<float>(roi_boxes[1]))) * spatial_scale;
        T roi_start_height = static_cast<T>(round(static_cast<float>(roi_boxes[2]))) * spatial_scale;
        T roi_end_width = static_cast<T>(round(static_cast<float>(roi_boxes[3]) + 1.)) * spatial_scale;
        T roi_end_height = static_cast<T>(round(static_cast<float>(roi_boxes[4]) + 1.)) * spatial_scale;

        // let min roi len and width bigger than 0.1
        T roi_width = max(roi_end_width - roi_start_width, 0.1);
        T roi_height = max(roi_end_height - roi_start_height, 0.1);

        // Compute bin_width and bin_height
        T bin_height = roi_height / static_cast<T>(pooled_height);
        T bin_width = roi_width / static_cast<T>(pooled_width);
        // compute pooling area's position
        int pooling_start_x = floor(static_cast<float>(static_cast<T>(height_offset_n) *
                              bin_height + roi_start_height));
        int pooling_start_y = floor(static_cast<float>(static_cast<T>(width_offset_n) *
                              bin_width + roi_start_width));
        int pooling_end_x = ceil(static_cast<float>(static_cast<T>(height_offset_n + 1) *
                              bin_height + roi_start_height));
        int pooling_end_y = ceil(static_cast<float>(static_cast<T>(width_offset_n + 1) *
                              bin_width + roi_start_width));
        // Add roi offsets and clip to input boundaries
        pooling_start_x = min(max(pooling_start_x, 0), feature_height);
        pooling_end_x = min(max(pooling_end_x, 0), feature_height);
        pooling_start_y = min(max(pooling_start_y, 0), feature_width);
        pooling_end_y = min(max(pooling_end_y, 0), feature_width);
        bool is_empty = (pooling_end_x <= pooling_start_x) || (pooling_end_y <= pooling_start_y);

        // roi_boxes[0] is roi_batch_index
        int c = mapping_channel[index];
        T* offset_bottom_diff = output_diff + (roi_batch_index * feature_channels + c) *
                                feature_height * feature_width;
        T bin_area = (pooling_end_x - pooling_start_x)*(pooling_end_y - pooling_start_y);
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
void PSROIPoolBackwardLauncher(const T* input_diff, const int* mapping_channel, const int batch_size,
    const int rois_number, const T spatial_scale, const int feature_channels,
    const int feature_height, const int feature_width, const int pooled_width,
    const int pooled_height, const int output_channels,
    T* output_diff, const T* roi_boxes, cudaStream_t stream) {

    size_t size_init = batch_size * feature_channels * feature_height * feature_width;
    PSROIPoolInitKernel<<<GET_BLOCKS(size_init), GET_THREADS, 0, stream>>>(size_init, output_diff);

    const int kThreadsPerBlock_ = 1024;
    const int output_size = output_channels * pooled_height * pooled_width * rois_number;
    cudaError_t err;

    PSROIPoolBackward<<<(output_size + kThreadsPerBlock_ - 1) / kThreadsPerBlock_, kThreadsPerBlock_, 0, stream>>>(
      output_size, input_diff, mapping_channel, spatial_scale, feature_height, feature_width, feature_channels,
      pooled_height, pooled_width, output_channels, output_diff, roi_boxes);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

template CUDA_LIB_EXPORT void PSROIPoolBackwardLauncher<float>(const float* input_diff, const int* mapping_channel,
                                                               const int batch_size, const int rois_number,
                                                               const float spatial_scale, const int feature_channels,
                                                               const int feature_height, const int feature_width,
                                                               const int pooled_width, const int pooled_height,
                                                               const int output_channels, float* output_diff,
                                                               const float* roi_boxes, cudaStream_t stream);

template CUDA_LIB_EXPORT void PSROIPoolBackwardLauncher<half>(const half* input_diff, const int* mapping_channel,
                                                              const int batch_size, const int rois_number,
                                                              const half spatial_scale, const int feature_channels,
                                                              const int feature_height, const int feature_width,
                                                              const int pooled_width, const int pooled_height,
                                                              const int output_channels, half* output_diff,
                                                              const half* roi_boxes, cudaStream_t stream);
