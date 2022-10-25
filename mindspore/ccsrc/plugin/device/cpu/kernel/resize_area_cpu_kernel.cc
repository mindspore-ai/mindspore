/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/resize_area_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/resize_area.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeAreaInputsNum = 2;
constexpr size_t kResizeAreaOutputsNum = 1;
int64_t ResizeAreaBound(int64_t val, int64_t limit) { return std::min(limit - 1, std::max(int64_t{0}, val)); }
}  // namespace

bool ResizeAreaCPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeAreaInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeAreaOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeArea>(base_operator);
  MS_ERROR_IF_NULL(kernel_ptr);
  align_corners_ = kernel_ptr->get_align_corners();
  return true;
}

int ResizeAreaCPUKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input0_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  batch_size_ = input0_shape[kIndex0];
  in_height_ = input0_shape[kIndex1];
  in_width_ = input0_shape[kIndex2];
  channels_ = input0_shape[kIndex3];

  auto output_shape = outputs[kIndex0]->GetDeviceShapeAdaptively();
  out_height_ = output_shape[kIndex1];
  out_width_ = output_shape[kIndex2];
  height_scale_ = Scaling(in_height_, out_height_, align_corners_);
  width_scale_ = Scaling(in_width_, out_width_, align_corners_);

  x_interps_.resize(out_width_);
  for (int64_t x = 0; x < out_width_; x++) {
    float transit_x0 = x * width_scale_;
    float transit_x1 = (x + 1) * width_scale_;
    int64_t v = std::floor(transit_x0);
    x_interps_[static_cast<size_t>(x)].start = v;
    x_interps_[static_cast<size_t>(x)].start_scale = (v + 1 > transit_x1 ? width_scale_ : v + 1 - transit_x0);
    v = std::ceil(transit_x1);
    x_interps_[static_cast<size_t>(x)].end = v;
    v = x_interps_[static_cast<size_t>(x)].end - 1;
    x_interps_[static_cast<size_t>(x)].end_minus_one_scale = (v + 1 > transit_x1 ? transit_x1 - v : 1.0);
  }
  return KRET_OK;
}

template <typename T>
bool ResizeAreaCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                                          const std::vector<ResizeAreaCachedInterpolation> &x_interps_) const {
  auto input_addr = static_cast<T *>(inputs[0]->addr);
  auto output_addr = static_cast<float *>(outputs[0]->addr);
  float scale = 1.0 / (height_scale_ * width_scale_);
  std::vector<float> y_scales;
  std::vector<const T *> y_ptrs;
  for (int64_t b = 0; b < batch_size_; ++b) {
    for (int64_t y = 0; y < out_height_; ++y) {
      y_scales.clear();
      y_ptrs.clear();
      const float transit_y0 = y * height_scale_;
      const float transit_y1 = (y + 1) * height_scale_;
      // The start and end height indices of all the cells that could
      // contribute to the target cell.
      const int64_t y_start = std::floor(transit_y0);
      const int64_t y_end = std::ceil(transit_y1);
      for (int64_t i = y_start; i < y_end; ++i) {
        float scale_y;
        if (i < transit_y0) {
          scale_y = (i + 1 > transit_y1 ? height_scale_ : i + 1 - transit_y0);
        } else {
          scale_y = (i + 1 > transit_y1 ? transit_y1 - i : 1.0);
        }
        y_scales.push_back(scale_y);
        y_ptrs.push_back(input_addr + (b * in_height_ * in_width_ * channels_ +
                                       ResizeAreaBound(i, in_height_) * in_width_ * channels_));
      }
      for (size_t x = 0; x < static_cast<size_t>(out_width_); ++x) {
        const ResizeAreaCachedInterpolation &x_interp = x_interps_[x];
        if (x_interp.needs_bounding) {
          ComputePatchSum<true>(scale, y_ptrs, y_scales, x_interp, output_addr);
        } else {
          ComputePatchSum<false>(scale, y_ptrs, y_scales, x_interp, output_addr);
        }
        output_addr += channels_;
      }
    }
  }
  return true;
}

// compute the value of the specific pxiel when the num of channels is not 3
template <bool NeedsXBounding, typename T>
void ResizeAreaCPUKernelMod::ComputePatchSum(float scale, const std::vector<const T *> &y_ptrs,
                                             const std::vector<float> &y_scales,
                                             const ResizeAreaCachedInterpolation &x_interp,
                                             float *output_patch_ptr) const {
#define BOUND_IF_NEEDED(x, y) (NeedsXBounding ? ResizeAreaBound(x, y) : (x))
  for (int64_t c = 0; c < channels_; ++c) {
    float sum = 0;
    for (size_t i = 0; i < y_ptrs.size(); ++i) {
      const T *ptr = y_ptrs[i];
      float scale_x = x_interp.start_scale;
      float sum_y =
        static_cast<float>(ptr[static_cast<size_t>(channels_ * BOUND_IF_NEEDED(x_interp.start, in_width_) + c)]) *
        scale_x;
      if (x_interp.start + 1 != x_interp.end) {
        for (int64_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
          sum_y += static_cast<float>(ptr[static_cast<size_t>(channels_ * BOUND_IF_NEEDED(x, in_width_) + c)]);
        }
        scale_x = x_interp.end_minus_one_scale;
        sum_y +=
          static_cast<float>(ptr[static_cast<size_t>(channels_ * BOUND_IF_NEEDED(x_interp.end - 1, in_width_) + c)]) *
          scale_x;
      }
      sum += sum_y * y_scales[i];
    }
    output_patch_ptr[static_cast<size_t>(c)] = sum * scale;
  }
#undef BOUND_IF_NEEDED
}

#define RESIZE_AREA_1D_CPU_REG(MS_T, T)                                                             \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32), \
    &ResizeAreaCPUKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ResizeAreaCPUKernelMod::ResizeAreaLaunchFunc>> ResizeAreaCPUKernelMod::func_list_ = {
  {RESIZE_AREA_1D_CPU_REG(kNumberTypeInt8, int8_t)},     {RESIZE_AREA_1D_CPU_REG(kNumberTypeInt16, int16_t)},
  {RESIZE_AREA_1D_CPU_REG(kNumberTypeInt32, int32_t)},   {RESIZE_AREA_1D_CPU_REG(kNumberTypeInt64, int64_t)},
  {RESIZE_AREA_1D_CPU_REG(kNumberTypeUInt8, uint8_t)},   {RESIZE_AREA_1D_CPU_REG(kNumberTypeUInt16, uint16_t)},
  {RESIZE_AREA_1D_CPU_REG(kNumberTypeFloat16, float16)}, {RESIZE_AREA_1D_CPU_REG(kNumberTypeFloat32, float)},
  {RESIZE_AREA_1D_CPU_REG(kNumberTypeFloat64, double)},
};

std::vector<KernelAttr> ResizeAreaCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ResizeAreaCPUKernelMod::ResizeAreaLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeArea, ResizeAreaCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
