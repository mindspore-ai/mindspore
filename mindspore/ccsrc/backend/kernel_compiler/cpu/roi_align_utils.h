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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROI_ALIGN_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROI_ALIGN_UTILS_H_
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {

constexpr int ROIS_COLS = 5;
constexpr size_t X_DIMS = 4;
constexpr int CHANNEL = 1;
constexpr int HEIGHT = 2;
constexpr int WIDTH = 3;

namespace roi {
template <typename T>
void bilinear_interpolate(const int height, const int width, T y, T x, int *x_low, int *y_low, int *x_high, int *y_high,
                          T *w1, T *w2, T *w3, T *w4) {
  constexpr float eps = 0.00007;
  const T ZERO = T(0.0);
  const T ONE = T(1.0);
  if (y < static_cast<T>(-1.0) || y > static_cast<T>(height) || x < static_cast<T>(-1.0) || x > static_cast<T>(width)) {
    *w1 = *w2 = *w3 = *w4 = static_cast<T>(0);
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }

  // low bounder is at least zero
  y = y <= ZERO ? ZERO : y;
  x = x <= ZERO ? ZERO : x;

  // top left point
  *y_low = (y <= static_cast<T>(eps) ? 0 : static_cast<int>(floor(y)));
  *x_low = (x <= static_cast<T>(eps) ? 0 : static_cast<int>(floor(x)));

  // bottom right point
  int h_limit = height - 1;
  if (*y_low >= h_limit) {
    *y_high = *y_low = h_limit;
    y = static_cast<T>(*y_low);
  } else {
    *y_high = *y_low + 1;
  }

  int w_limit = width - 1;
  if (*x_low >= w_limit) {
    *x_high = *x_low = w_limit;
    x = static_cast<T>(*x_low);
  } else {
    *x_high = *x_low + 1;
  }

  // distance to nearest points
  T lx, ly, hx, hy;
  ly = y - static_cast<T>(*y_low), lx = x - static_cast<T>(*x_low);
  hy = ONE - ly, hx = ONE - lx;

  // weight is evaluated by the distance to point away.
  //   the closer to point home, the more weight, the farther to point away.
  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;
  return;
}

template <typename T>
void bin_box(int thread_idx, const T *roi_boxes, int roi_cols, const T spatial_scale, const int sample_num,
             int roi_end_mode, const int channels, const int height, const int width, const int pooled_height,
             const int pooled_width, int *offset, int *n, int *c, int *ph, int *pw, int *roi_bin_grid_h,
             int *roi_bin_grid_w, T *bin_size_h, T *bin_size_w, T *roi_start_h, T *roi_start_w) {
  MS_EXCEPTION_IF_ZERO("pooled_height", pooled_height);
  MS_EXCEPTION_IF_ZERO("pooled_width", pooled_width);
  MS_EXCEPTION_IF_ZERO("channels", channels);
  constexpr int START_W = 0;
  constexpr int START_H = 1;
  constexpr int END_W = 2;
  constexpr int END_H = 3;
  constexpr float eps = 0.00007;
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
  if (roi_cols == ROIS_COLS) {
    roi_batch_ind = FloatToInt(rint(static_cast<float>(roi_box[0]) + eps));
    roi_box++;
  }

  // Scale and shift ROI
  *roi_start_w = roi_box[START_W] * spatial_scale;
  *roi_start_h = roi_box[START_H] * spatial_scale;
  T roi_end_w = (roi_box[END_W] + static_cast<T>(roi_end_mode)) * spatial_scale;
  T roi_end_h = (roi_box[END_H] + static_cast<T>(roi_end_mode)) * spatial_scale;

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
  *roi_bin_grid_h = (sample_num > 0) ? sample_num : static_cast<int>(floor(roi_height / static_cast<T>(pooled_height)));
  *roi_bin_grid_w = (sample_num > 0) ? sample_num : static_cast<int>(floor(roi_width / static_cast<T>(pooled_width)));
  return;
}

}  // namespace roi
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROI_ALIGN_UTILS_H_
