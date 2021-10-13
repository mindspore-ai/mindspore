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

#include "backend/kernel_compiler/cpu/roi_align_utils.h"
#include "backend/kernel_compiler/cpu/roi_align_grad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T, typename U>
void AtomicAddTask(T *const address, const T val) {
  auto *address_as_ull = reinterpret_cast<U *>(address);
  U old = *address_as_ull;
  U assumed = U(0);
  T desired = T(0);
  do {
    assumed = old;
    T *assumed_t = reinterpret_cast<T *>(&assumed);
    U *desired_u = reinterpret_cast<U *>(&desired);
    desired = *assumed_t + static_cast<T>(val);
    old = __sync_val_compare_and_swap(address_as_ull, assumed, *desired_u);
  } while (assumed != old);
}

template <typename T>
void AtomicAdd(T *const address, const T val) {
  switch (sizeof(T)) {
    case sizeof(int8_t): {
      AtomicAddTask<T, int8_t>(address, val);
      break;
    }
    case sizeof(int16_t): {
      AtomicAddTask<T, int16_t>(address, val);
      break;
    }
    case sizeof(int32_t): {
      AtomicAddTask<T, int32_t>(address, val);
      break;
    }
    case sizeof(int64_t): {
      AtomicAddTask<T, int64_t>(address, val);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Unsupported datatype.";
  }
}

template <typename T>
void ROIAlignGradCPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  //  Get the number of the input args
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != INPUT_NUM) {
    MS_LOG(ERROR) << "Input number is: " << input_num << ", but ROIAlignGrad needs 2 inputs.";
  }

  //  Get the number of the output args
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != OUTPUT_NUM) {
    MS_LOG(ERROR) << "Output number is: " << output_num << ", but ROIAlignGrad needs 1 output.";
  }

  //  Get the input shapes
  auto dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto dy_shape_size = dy_shape.size();
  if (dy_shape_size != DY_DIMS) {
    MS_LOG(ERROR) << "dy shape size is " << dy_shape_size << ", but should be 4.";
  }
}

template <typename T>
void ROIAlignGradCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  CheckParam(kernel_node);

  auto rois_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  roi_rows_ = SizeToInt(rois_shape[0]);
  roi_cols_ = SizeToInt(rois_shape[1]);

  std::vector<int64_t> xdiff_shape_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "xdiff_shape");
  (void)std::transform(xdiff_shape_me.begin(), xdiff_shape_me.end(), std::back_inserter(xdiff_shape_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  pooled_height_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "pooled_height"));
  pooled_width_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "pooled_width"));
  spatial_scale_ = static_cast<T>(AnfAlgo::GetNodeAttr<float>(kernel_node, "spatial_scale"));
  sample_num_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "sample_num"));
  roi_end_mode_ = 1;

  batch_size_ = xdiff_shape_[BATCH];
  channels_ = xdiff_shape_[CHANNEL];
  height_ = xdiff_shape_[HEIGHT];
  width_ = xdiff_shape_[WIDTH];
}

template <typename T>
bool ROIAlignGradCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  const T *dy = reinterpret_cast<T *>(inputs[0]->addr);
  const T *rois = reinterpret_cast<T *>(inputs[1]->addr);
  T *dx = reinterpret_cast<T *>(outputs[0]->addr);

  int size_init = batch_size_ * channels_ * height_ * width_;
  auto task1 = [this, &dx](size_t start, size_t end) {
    const T ZERO = T(0.0);
    for (size_t thread_idx = start; thread_idx < end; thread_idx++) {
      dx[thread_idx] = ZERO;
    }
  };
  CPUKernelUtils::ParallelFor(task1, IntToSize(size_init));

  int elem_num = roi_rows_ * channels_ * pooled_height_ * pooled_width_;
  auto task2 = [this, &dy, &rois, &dx](size_t start, size_t end) {
    const T OFFSET = T(0.001);
    for (size_t thread_idx = start; thread_idx < end; thread_idx++) {
      int n = SizeToInt(thread_idx) / pooled_width_ / pooled_height_ / channels_;
      const T *roi_box = rois + n * roi_cols_;
      if (roi_box[1] < OFFSET && roi_box[3] < OFFSET && roi_box[1] > -OFFSET && roi_box[3] > -OFFSET) {
        continue;
      }
      int offset = -1;
      int c, ph, pw, roi_bin_grid_h, roi_bin_grid_w;
      T bin_size_h, bin_size_w, roi_start_h, roi_start_w;

      roi::bin_box(SizeToInt(thread_idx), rois, roi_cols_, spatial_scale_, sample_num_, roi_end_mode_, channels_,
                   height_, width_, pooled_height_, pooled_width_, &offset, &n, &c, &ph, &pw, &roi_bin_grid_h,
                   &roi_bin_grid_w, &bin_size_h, &bin_size_w, &roi_start_h, &roi_start_w);

      // (n, c, ph, pw) is the base param of pooled map
      const T count_points_in_grid_cell = static_cast<T>(roi_bin_grid_h) * static_cast<T>(roi_bin_grid_w);

      int top_offset = (n * channels_ + c) * pooled_height_ * pooled_width_;
      const T *offset_top_diff = dy + top_offset;
      const T top_diff_this_bin = offset_top_diff[ph * pooled_width_ + pw];

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
          roi::bilinear_interpolate(height_, width_, y, x, &x_low, &y_low, &x_high, &y_high, &w1, &w2, &w3, &w4);
          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && y_low < height_ && y_high < height_ &&
              x_low < width_ && x_high < width_) {
            T g1 = top_diff_this_bin * w1 / count_points_in_grid_cell;
            T g2 = top_diff_this_bin * w2 / count_points_in_grid_cell;
            T g3 = top_diff_this_bin * w3 / count_points_in_grid_cell;
            T g4 = top_diff_this_bin * w4 / count_points_in_grid_cell;

            T *dx_1 = dx + offset + y_low * width_ + x_low;
            T *dx_2 = dx + offset + y_low * width_ + x_high;
            T *dx_3 = dx + offset + y_high * width_ + x_low;
            T *dx_4 = dx + offset + y_high * width_ + x_high;

            AtomicAdd(dx_1, g1);
            AtomicAdd(dx_2, g2);
            AtomicAdd(dx_3, g3);
            AtomicAdd(dx_4, g4);
          }
        }
      }
    }
  };
  CPUKernelUtils::ParallelFor(task2, IntToSize(elem_num));
  return true;
}
}  // namespace kernel
}  // namespace mindspore
