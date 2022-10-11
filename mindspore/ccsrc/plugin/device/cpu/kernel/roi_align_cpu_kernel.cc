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
#include "plugin/device/cpu/kernel/roi_align_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
void bilinear_interpolate(const int height, const int width, T y, T x, int *x_low, int *y_low, int *x_high, int *y_high,
                          T *w1, T *w2, T *w3, T *w4) {
  constexpr float kEps = 0.00007;
  const T kZero = T(0.0);
  const T kOne = T(1.0);
  const T kMinusOne = static_cast<T>(-1.0);
  if (y < kMinusOne || y > static_cast<T>(height) || x < kMinusOne || x > static_cast<T>(width)) {
    *w1 = *w2 = *w3 = *w4 = static_cast<T>(0);
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }

  // low bounder is at least zero
  y = y <= kZero ? kZero : y;
  x = x <= kZero ? kZero : x;

  // top left point
  *y_low = (y <= static_cast<T>(kEps) ? 0 : static_cast<int>(floor(y)));
  *x_low = (x <= static_cast<T>(kEps) ? 0 : static_cast<int>(floor(x)));

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
  hy = kOne - ly, hx = kOne - lx;

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
  constexpr float kEps = 0.00007;
  constexpr int kStartW = 0;
  constexpr int kStartH = 1;
  constexpr int kEndW = 2;
  constexpr int kEndH = 3;
  constexpr int kRoisCols = 5;
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
  if (roi_cols == kRoisCols) {
    roi_batch_ind = FloatToInt(rintf(static_cast<float>(roi_box[0]) + kEps));
    roi_box++;
  }

  // Scale and shift ROI
  *roi_start_w = roi_box[kStartW] * spatial_scale;
  *roi_start_h = roi_box[kStartH] * spatial_scale;
  T roi_end_w = (roi_box[kEndW] + static_cast<T>(roi_end_mode)) * spatial_scale;
  T roi_end_h = (roi_box[kEndH] + static_cast<T>(roi_end_mode)) * spatial_scale;

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
}  // namespace

bool ROIAlignCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  // Check input and output numbers
  kernel_name_ = base_operator->name();
  constexpr size_t kInputSize = 2;
  constexpr size_t kOutputSize = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  // Get primitive args
  auto op = std::dynamic_pointer_cast<ops::ROIAlign>(base_operator);
  pooled_height_ = LongToInt(op->get_pooled_height());
  pooled_width_ = LongToInt(op->get_pooled_width());
  spatial_scale_ = op->get_spatial_scale();
  sample_num_ = LongToInt(op->get_sample_num());
  roi_end_mode_ = LongToInt(op->get_roi_end_mode());

  MS_EXCEPTION_IF_ZERO("pooled_height", pooled_height_);
  MS_EXCEPTION_IF_ZERO("pooled_width", pooled_width_);
  return true;
}

int ROIAlignCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  //  Get the input shapes
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  auto rois_shape = inputs[kIndex1]->GetShapeVector();
  constexpr size_t kFeatureDims = 4;
  constexpr size_t kRoisDims = 2;
  if (x_shape.size() > kFeatureDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'features' cannot be greater than 4, but got "
                  << x_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  if (rois_shape.size() != kRoisDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'rois' must be equal to 2, but got "
                  << rois_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  // Calculate the sizes of inputs and output
  auto x_type_size = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  auto rois_type_size = abstract::TypeIdSize(inputs[kIndex1]->GetDtype());
  x_size_ = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies{}) * x_type_size;
  rois_size_ = std::accumulate(rois_shape.begin(), rois_shape.end(), 1, std::multiplies{}) * rois_type_size;
  output_size_ = rois_shape[kIndex0] * x_shape[kIndex1] * pooled_height_ * pooled_width_ * x_type_size;

  channels_ = LongToInt(x_shape[kIndex1]);
  MS_EXCEPTION_IF_ZERO("channels", channels_);
  height_ = LongToInt(x_shape[kIndex2]);
  width_ = LongToInt(x_shape[kIndex3]);

  roi_rows_ = LongToInt(rois_shape[kIndex0]);
  roi_cols_ = LongToInt(rois_shape[kIndex1]);

  ResetResource();
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool ROIAlignCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const T *input = reinterpret_cast<T *>(inputs[0]->addr);
  const T *rois = reinterpret_cast<T *>(inputs[1]->addr);
  auto out_data = reinterpret_cast<T *>(outputs[0]->addr);

  size_t elem_num = IntToSize(roi_rows_ * channels_ * pooled_height_ * pooled_width_);
  auto task = [this, &input, &rois, &out_data](size_t start, size_t end) {
    const T kOffset = T(0.001);
    const T kZero = T(0.0);
    const T spatial_scale = static_cast<T>(spatial_scale_);
    for (size_t thread_idx = start; thread_idx < end; thread_idx++) {
      int n = SizeToInt(thread_idx) / pooled_width_ / pooled_height_ / channels_;
      const T *roi_box = rois + n * roi_cols_;
      if (roi_box[1] < kOffset && roi_box[3] < kOffset && roi_box[1] > -kOffset && roi_box[3] > -kOffset) {
        continue;
      }
      int offset = -1;
      int c, ph, pw, roi_bin_grid_h, roi_bin_grid_w;
      T bin_size_h, bin_size_w, roi_start_h, roi_start_w;

      bin_box(SizeToInt(thread_idx), rois, roi_cols_, spatial_scale, sample_num_, roi_end_mode_, channels_, height_,
              width_, pooled_height_, pooled_width_, &offset, &n, &c, &ph, &pw, &roi_bin_grid_h, &roi_bin_grid_w,
              &bin_size_h, &bin_size_w, &roi_start_h, &roi_start_w);

      // (n, c, ph, pw) is the base param of pooled map
      const T count_points_in_grid_cell = static_cast<T>(roi_bin_grid_h) * static_cast<T>(roi_bin_grid_w);

      T accumulate_val = kZero;
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
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
  return true;
}

const ROIAlignCpuKernelMod::FuncList &ROIAlignCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ROIAlignCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ROIAlignCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ROIAlignCpuKernelMod::LaunchKernel<float16>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ROIAlign, ROIAlignCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
