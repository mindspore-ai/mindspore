/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <utility>
#include <memory>
#include "plugin/device/cpu/kernel/roi_align_grad_cpu_kernel.h"
#include "plugin/device/cpu/kernel/atomic_add.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
class ROIAlignGradCpuKernelFunc : public DeprecatedCpuKernelFunc {
 public:
  ROIAlignGradCpuKernelFunc() = default;
  ~ROIAlignGradCpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
               const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);

  void bilinear_interpolate(const int height, const int width, T y, T x, int *x_low, int *y_low, int *x_high,
                            int *y_high, T *w1, T *w2, T *w3, T *w4) const;

  void bin_box(int thread_idx, const T *roi_boxes, int roi_cols, const T spatial_scale, const int sample_num,
               int roi_end_mode, const int channels, const int height, const int width, const int pooled_height,
               const int pooled_width, int *offset, int *n, int *c, int *ph, int *pw, int *roi_bin_grid_h,
               int *roi_bin_grid_w, T *bin_size_h, T *bin_size_w, T *roi_start_h, T *roi_start_w) const;

  std::vector<int> xdiff_shape_;
  int pooled_height_{0};
  int pooled_width_{0};
  T spatial_scale_{0.0};
  int sample_num_{0};
  int roi_end_mode_{0};

  int roi_rows_{0};
  int roi_cols_{0};
  int batch_size_{0};
  int channels_{0};
  int height_{0};
  int width_{0};

  std::string kernel_name_;
};

#ifndef _MSC_VER
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
#else
template <typename T, typename U>
void AtomicAddTask(T *const address, const T val) {
  *address = (*address) + val;
}
#endif

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
      MS_LOG(EXCEPTION) << "For 'ROIAlignGrad', the dtype " << typeid(T).name() << " is unsupported.";
  }
}

template <typename T>
void ROIAlignGradCpuKernelFunc<T>::CheckParam(const CNodePtr &kernel_node) {
  //  Get the number of the input args
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != INPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num
                  << " input(s).";
  }

  //  Get the number of the output args
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num
                  << " output(s).";
  }

  //  Get the input shapes
  auto dy_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto dy_shape_size = dy_shape.size();
  if (dy_shape_size != DY_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of output must be 4-D, but got " << dy_shape_size
                  << "-D.";
  }
}

template <typename T>
void ROIAlignGradCpuKernelFunc<T>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  CheckParam(kernel_node);

  auto rois_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  roi_rows_ = LongToInt(rois_shape[0]);
  roi_cols_ = LongToInt(rois_shape[1]);

  std::vector<int64_t> xdiff_shape_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "xdiff_shape");
  (void)std::transform(xdiff_shape_me.begin(), xdiff_shape_me.end(), std::back_inserter(xdiff_shape_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  pooled_height_ = static_cast<int>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "pooled_height"));
  pooled_width_ = static_cast<int>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "pooled_width"));
  spatial_scale_ = static_cast<T>(common::AnfAlgo::GetNodeAttr<float>(kernel_node, "spatial_scale"));
  sample_num_ = static_cast<int>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "sample_num"));
  roi_end_mode_ = 1;

  batch_size_ = xdiff_shape_[BATCH];
  channels_ = xdiff_shape_[CHANNEL];
  height_ = xdiff_shape_[HEIGHT];
  width_ = xdiff_shape_[WIDTH];
}

template <typename T>
bool ROIAlignGradCpuKernelFunc<T>::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
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
  ParallelLaunchAutoSearch(task1, IntToSize(size_init), this, &parallel_search_info_);

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

      bin_box(SizeToInt(thread_idx), rois, roi_cols_, spatial_scale_, sample_num_, roi_end_mode_, channels_, height_,
              width_, pooled_height_, pooled_width_, &offset, &n, &c, &ph, &pw, &roi_bin_grid_h, &roi_bin_grid_w,
              &bin_size_h, &bin_size_w, &roi_start_h, &roi_start_w);

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
          bilinear_interpolate(height_, width_, y, x, &x_low, &y_low, &x_high, &y_high, &w1, &w2, &w3, &w4);
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
  ParallelLaunchAutoSearch(task2, IntToSize(elem_num), this, &parallel_search_info_);
  return true;
}

template <typename T>
void ROIAlignGradCpuKernelFunc<T>::bilinear_interpolate(const int height, const int width, T y, T x, int *x_low,
                                                        int *y_low, int *x_high, int *y_high, T *w1, T *w2, T *w3,
                                                        T *w4) const {
  constexpr float eps = 0.00007;
  const T ZERO = T(0.0);
  const T ONE = T(1.0);
  const T NEG_ONE = static_cast<T>(-1.0);
  if (y < NEG_ONE || y > static_cast<T>(height) || x < NEG_ONE || x > static_cast<T>(width)) {
    *w1 = *w2 = *w3 = *w4 = ZERO;
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
  hy = ONE - ly, hx = ONE - lx;

  // weight is evaluated by the distance to point away.
  //   the closer to point home, the more weight, the farther to point away.
  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;
  return;
}

template <typename T>
void ROIAlignGradCpuKernelFunc<T>::bin_box(int thread_idx, const T *roi_boxes, int roi_cols, const T spatial_scale,
                                           const int sample_num, int roi_end_mode, const int channels, const int height,
                                           const int width, const int pooled_height, const int pooled_width,
                                           int *offset, int *n, int *c, int *ph, int *pw, int *roi_bin_grid_h,
                                           int *roi_bin_grid_w, T *bin_size_h, T *bin_size_w, T *roi_start_h,
                                           T *roi_start_w) const {
  constexpr float eps = 0.00007;
  constexpr int START_W = 0;
  constexpr int START_H = 1;
  constexpr int END_W = 2;
  constexpr int END_H = 3;
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
    roi_batch_ind = FloatToInt(rintf(static_cast<float>(roi_box[0]) + eps));
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

template <typename T>
std::shared_ptr<DeprecatedCpuKernelFunc> SpecializeROIAlignGradFunc() {
  return std::make_shared<ROIAlignGradCpuKernelFunc<T>>();
}
using SpecializeROIAlignGradFuncCreator = std::function<std::shared_ptr<DeprecatedCpuKernelFunc>()>;
static std::vector<std::pair<KernelAttr, SpecializeROIAlignGradFuncCreator>> kernel_attr_list = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   SpecializeROIAlignGradFunc<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   SpecializeROIAlignGradFunc<float16>}};
}  // namespace

void ROIAlignGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "ROIAlignGrad does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list[index].second();
  func_obj_->InitFunc(kernel_node);
}

std::vector<KernelAttr> ROIAlignGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr_list.begin(), kernel_attr_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpecializeROIAlignGradFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ROIAlignGrad, ROIAlignGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
