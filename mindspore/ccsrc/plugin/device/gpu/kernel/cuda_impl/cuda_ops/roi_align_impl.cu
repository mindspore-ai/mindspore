/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "roi_align_impl.cuh"
#include "util.cuh"

inline __device__ int roi_cast_int(float x) { return __float2int_rd(x); }
inline __device__ int roi_cast_int(half x) { return __half2int_rd(x); }
inline __device__ int roi_round_int(float x) { return __float2int_rn(x + 0.00007); }
inline __device__ int roi_round_int(half x) { return __half2int_rn(x + static_cast<half>(0.00007)); }

template <typename T>
__device__ void bilinear_interpolate(const int height, const int width, T y, T x, int *x_low, int *y_low, int *x_high,
                                     int *y_high, T *w1, T *w2, T *w3, T *w4) {
  // return 0 if out of map boundary
  constexpr float eps = 0.00007;
  if (y < static_cast<T>(-1.0) || y > static_cast<T>(height) || x < static_cast<T>(-1.0) || x > static_cast<T>(width)) {
    *w1 = *w2 = *w3 = *w4 = 0;
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }

  // low bounder is at least zero
  y = y <= static_cast<T>(.0) ? static_cast<T>(.0) : y;
  x = x <= static_cast<T>(.0) ? static_cast<T>(.0) : x;

  // top left point
  *y_low = (y <= static_cast<T>(eps) ? 0 : roi_cast_int(y));
  *x_low = (x <= static_cast<T>(eps) ? 0 : roi_cast_int(x));

  // bottom right point
  if (*y_low >= height - 1) {
    *y_high = *y_low = height - 1;
    y = static_cast<T>(*y_low);
  } else {
    *y_high = *y_low + 1;
  }

  if (*x_low >= width - 1) {
    *x_high = *x_low = width - 1;
    x = static_cast<T>(*x_low);
  } else {
    *x_high = *x_low + 1;
  }

  // distance to nearest points
  T lx, ly, hx, hy;
  ly = y - static_cast<T>(*y_low), lx = x - static_cast<T>(*x_low);
  hy = static_cast<T>(1.) - ly, hx = static_cast<T>(1.) - lx;

  // weight is evaluated by the distance to point away.
  //   the closer to point home, the more weight, the farther to point away.
  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;
}

template <typename T>
__device__ void bin_box(int thread_idx, const T *roi_boxes, int roi_cols, const T spatial_scale, const int sample_num,
                        int roi_end_mode, const int channels, const int height, const int width,
                        const int pooled_height, const int pooled_width, int *offset, int *n, int *c, int *ph, int *pw,
                        int *roi_bin_grid_h, int *roi_bin_grid_w, T *bin_size_h, T *bin_size_w, T *roi_start_h,
                        T *roi_start_w) {
  // (n, c, ph, pw) is the base param of pooled map
  *pw = thread_idx % pooled_width;
  *ph = (thread_idx / pooled_width) % pooled_height;
  *c = (thread_idx / pooled_width / pooled_height) % channels;
  *n = thread_idx / pooled_width / pooled_height / channels;

  // Roi has
  //   1. 4 points, or
  //   2. indicator + 4 points (1 + 4)
  const T *roi_box = roi_boxes + (*n) * roi_cols;
  int roi_batch_ind = 0;
  if (roi_cols == 5) {
    roi_batch_ind = roi_round_int(roi_box[0]);
    roi_box++;
  }

  // Scale and shift ROI
  *roi_start_w = roi_box[0] * spatial_scale;
  *roi_start_h = roi_box[1] * spatial_scale;
  T roi_end_w = (roi_box[2] + static_cast<T>(roi_end_mode)) * spatial_scale;
  T roi_end_h = (roi_box[3] + static_cast<T>(roi_end_mode)) * spatial_scale;

  // New ROI height/width
  T roi_width = roi_end_w - (*roi_start_w);
  T roi_height = roi_end_h - (*roi_start_h);

  if (roi_end_mode == 0) {  // backward compatibility
    // Force malformed ROIs to be 1x1
    roi_width = roi_width > static_cast<T>(1.0) ? roi_width : static_cast<T>(1.0);
    roi_height = roi_height > static_cast<T>(1.0) ? roi_height : static_cast<T>(1.0);
  }

  // ratio of roi / pooled
  *bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  *bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  *offset = (roi_batch_ind * channels + (*c)) * height * width;

  // grid (int) by Sample ratio if defined, otherwise by pooled H/W
  *roi_bin_grid_h = (sample_num > 0) ? sample_num : roi_cast_int(roi_height / static_cast<T>(pooled_height));
  *roi_bin_grid_w = (sample_num > 0) ? sample_num : roi_cast_int(roi_width / static_cast<T>(pooled_width));
}

template <typename T>
__global__ void ROIAlignKernel(size_t size, const T *input, const T *roi_boxes, int roi_cols, T *out_data,
                               const T spatial_scale, const int sample_num, int roi_end_mode, const int channels,
                               const int height, const int width, const int pooled_height, const int pooled_width) {
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < size;
       thread_idx += blockDim.x * gridDim.x) {
    int n = thread_idx / pooled_width / pooled_height / channels;
    const T *roi_box = roi_boxes + n * roi_cols;
    // Skip if roi box is a line
    if (roi_box[1] < static_cast<T>(0.001) && roi_box[3] < static_cast<T>(0.001) &&
        roi_box[1] > static_cast<T>(-0.001) && roi_box[3] > static_cast<T>(-0.001)) {
      continue;
    }

    int offset = -1;
    int c, ph, pw, roi_bin_grid_h, roi_bin_grid_w;
    T bin_size_h, bin_size_w, roi_start_h, roi_start_w;

    bin_box(thread_idx, roi_boxes, roi_cols, spatial_scale, sample_num, roi_end_mode, channels, height, width,
            pooled_height, pooled_width, &offset, &n, &c, &ph, &pw, &roi_bin_grid_h, &roi_bin_grid_w, &bin_size_h,
            &bin_size_w, &roi_start_h, &roi_start_w);

    // (n, c, ph, pw) is the base param of pooled map
    const T count_points_in_grid_cell = roi_bin_grid_h * roi_bin_grid_w;

    T accumulate_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      // Shift half point RIGHT for y / x,  while previous scaled roi shift half point LEFT
      const T y = roi_start_h + static_cast<T>(ph) * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + static_cast<T>(pw) * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
        // bilinear interpolate by shifted y / x
        // calculate bilinear interpolation
        int x_low = 0, y_low = 0, x_high = 0, y_high = 0;
        T w1, w2, w3, w4;
        bilinear_interpolate(height, width, y, x, &x_low, &y_low, &x_high, &y_high, &w1, &w2, &w3, &w4);
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && y_low < height && y_high < height &&
            x_low < width && x_high < width) {
          T v1 = input[offset + y_low * width + x_low];
          T v2 = input[offset + y_low * width + x_high];
          T v3 = input[offset + y_high * width + x_low];
          T v4 = input[offset + y_high * width + x_high];

          T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
          accumulate_val += val;
        }
      }
    }
    accumulate_val /= count_points_in_grid_cell;

    out_data[thread_idx] = accumulate_val;
  }
}

template <typename T>
cudaError_t ROIAlign(const T *x, const T *roi_boxes, int roi_rows, int roi_cols, T *out_data, const T spatial_scale,
                     const int sample_num, int roi_end_mode, const int channels, const int height, const int width,
                     const int pooled_height, const int pooled_width, const uint32_t &device_id,
                     cudaStream_t cuda_stream) {
  size_t size = roi_rows * channels * pooled_height * pooled_width;
  ROIAlignKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, x, roi_boxes, roi_cols, out_data, spatial_scale, sample_num, roi_end_mode, channels, height, width,
    pooled_height, pooled_width);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ROIAlign<float>(const float *x, const float *roi_boxes, int roi_rows, int roi_cols,
                                                     float *out_data, const float spatial_scale, const int sample_num,
                                                     int roi_end_mode, const int channels, const int height,
                                                     const int width, const int pooled_height, const int pooled_width,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ROIAlign<half>(const half *x, const half *roi_boxes, int roi_rows, int roi_cols,
                                                    half *out_data, const half spatial_scale, const int sample_num,
                                                    int roi_end_mode, const int channels, const int height,
                                                    const int width, const int pooled_height, const int pooled_width,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
__global__ void ROIAlignGradInitKernel(size_t size_init, T *dx) {
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < size_init;
       thread_idx += blockDim.x * gridDim.x) {
    dx[thread_idx] = static_cast<T>(.0);
  }
}

template <typename T>
__global__ void ROIAlignGradKernel(size_t size, const T *dy, const T *roi_boxes, int roi_cols, T *dx,
                                   const T spatial_scale, const int sample_num, int roi_end_mode, const int channels,
                                   const int height, const int width, const int pooled_height, const int pooled_width) {
  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < size;
       thread_idx += blockDim.x * gridDim.x) {
    int n = thread_idx / pooled_width / pooled_height / channels;
    const T *roi_box = roi_boxes + n * roi_cols;
    if (roi_box[1] < static_cast<T>(0.001) && roi_box[3] < static_cast<T>(0.001) &&
        roi_box[1] > static_cast<T>(-0.001) && roi_box[3] > static_cast<T>(-0.001)) {
      continue;
    }

    int offset = -1;
    int c, ph, pw, roi_bin_grid_h, roi_bin_grid_w;
    T bin_size_h, bin_size_w, roi_start_h, roi_start_w;

    bin_box(thread_idx, roi_boxes, roi_cols, spatial_scale, sample_num, roi_end_mode, channels, height, width,
            pooled_height, pooled_width, &offset, &n, &c, &ph, &pw, &roi_bin_grid_h, &roi_bin_grid_w, &bin_size_h,
            &bin_size_w, &roi_start_h, &roi_start_w);

    // (n, c, ph, pw) is the base param of pooled map
    const T count_points_in_grid_cell = roi_bin_grid_h * roi_bin_grid_w;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T *offset_top_diff = dy + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      // Shift half point RIGHT for y / x,  while previous scaled roi shift half point LEFT
      const T y = roi_start_h + static_cast<T>(ph) * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + static_cast<T>(pw) * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
        // bilinear interpolate by shifted y / x
        // calculate bilinear interpolation
        int x_low = 0, y_low = 0, x_high = 0, y_high = 0;
        T w1, w2, w3, w4;
        bilinear_interpolate(height, width, y, x, &x_low, &y_low, &x_high, &y_high, &w1, &w2, &w3, &w4);
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && y_low < height && y_high < height &&
            x_low < width && x_high < width) {
          T g1 = top_diff_this_bin * w1 / count_points_in_grid_cell;
          T g2 = top_diff_this_bin * w2 / count_points_in_grid_cell;
          T g3 = top_diff_this_bin * w3 / count_points_in_grid_cell;
          T g4 = top_diff_this_bin * w4 / count_points_in_grid_cell;

          T *dx_1 = dx + offset + y_low * width + x_low;
          T *dx_2 = dx + offset + y_low * width + x_high;
          T *dx_3 = dx + offset + y_high * width + x_low;
          T *dx_4 = dx + offset + y_high * width + x_high;

          MsAtomicAdd(dx_1, g1);
          MsAtomicAdd(dx_2, g2);
          MsAtomicAdd(dx_3, g3);
          MsAtomicAdd(dx_4, g4);
        }
      }
    }
  }
}

template <typename T>
cudaError_t ROIAlignGrad(const T *dy, const T *roi_boxes, int batch_size, int roi_rows, int roi_cols, T *dx,
                         const T spatial_scale, const int sample_num, int roi_end_mode, const int channels,
                         const int height, const int width, const int pooled_height, const int pooled_width,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size_init = batch_size * channels * height * width;
  ROIAlignGradInitKernel<<<CUDA_BLOCKS(device_id, size_init), CUDA_THREADS(0), 0, cuda_stream>>>(size_init, dx);

  size_t size = roi_rows * channels * pooled_height * pooled_width;
  ROIAlignGradKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(0), 0, cuda_stream>>>(
    size, dy, roi_boxes, roi_cols, dx, spatial_scale, sample_num, roi_end_mode, channels, height, width, pooled_height,
    pooled_width);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ROIAlignGrad<float>(const float *dy, const float *roi_boxes, int batch_size,
                                                         int roi_rows, int roi_cols, float *dx,
                                                         const float spatial_scale, const int sample_num,
                                                         int roi_end_mode, const int channels, const int height,
                                                         const int width, const int pooled_height,
                                                         const int pooled_width, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ROIAlignGrad<half>(const half *dy, const half *roi_boxes, int batch_size,
                                                        int roi_rows, int roi_cols, half *dx, const half spatial_scale,
                                                        const int sample_num, int roi_end_mode, const int channels,
                                                        const int height, const int width, const int pooled_height,
                                                        const int pooled_width, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
