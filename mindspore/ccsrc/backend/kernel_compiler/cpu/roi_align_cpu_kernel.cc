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

#include "backend/kernel_compiler/cpu/roi_align_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/roi_align_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 2;
constexpr size_t kOutputSize = 1;
}  //  namespace

template <typename T>
void ROIAlignCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  //  Get the input shapes
  auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto rois_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

  auto x_shape_size = x_shape.size();
  if (x_shape_size != X_DIMS) {
    MS_LOG(ERROR) << "x shape size is " << x_shape_size << ", but should be 4.";
  }

  channels_ = SizeToInt(x_shape[CHANNEL]);
  height_ = SizeToInt(x_shape[HEIGHT]);
  width_ = SizeToInt(x_shape[WIDTH]);

  roi_rows_ = SizeToInt(rois_shape[0]);
  roi_cols_ = SizeToInt(rois_shape[1]);

  pooled_height_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "pooled_height"));
  pooled_width_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "pooled_width"));
  spatial_scale_ = static_cast<T>(AnfAlgo::GetNodeAttr<float>(kernel_node, "spatial_scale"));
  sample_num_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "sample_num"));
  roi_end_mode_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "roi_end_mode"));
}

template <typename T>
bool ROIAlignCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  const T *input = reinterpret_cast<T *>(inputs[0]->addr);
  const T *rois = reinterpret_cast<T *>(inputs[1]->addr);
  auto out_data = reinterpret_cast<T *>(outputs[0]->addr);

  size_t elem_num = IntToSize(roi_rows_ * channels_ * pooled_height_ * pooled_width_);
  auto task = [this, &input, &rois, &out_data](size_t start, size_t end) {
    const T OFFSET = T(0.001);
    const T ZERO = T(0.0);
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

      T accumulate_val = ZERO;
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
            T v1 = input[offset + y_low * width_ + x_low];
            T v2 = input[offset + y_low * width_ + x_high];
            T v3 = input[offset + y_high * width_ + x_low];
            T v4 = input[offset + y_high * width_ + x_high];

            T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
            accumulate_val += val;
          }
        }
      }
      accumulate_val /= count_points_in_grid_cell;

      out_data[thread_idx] = accumulate_val;
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num);

  return true;
}

template <typename T>
void ROIAlignCPUKernel<T>::CheckParam(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kInputSize) {
    MS_LOG(EXCEPTION) << "Input number is: " << inputs.size() << ", but ROIAlign needs " << kInputSize << " inputs.";
  }

  if (outputs.size() != kOutputSize) {
    MS_LOG(EXCEPTION) << "Output number is: " << outputs.size() << ", but ROIAlign needs " << kOutputSize << "outputs.";
  }
}
}  // namespace kernel
}  // namespace mindspore
