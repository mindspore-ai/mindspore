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

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeAreaInputsNum = 2;
constexpr size_t kResizeAreaOutputsNum = 1;
constexpr size_t kResizeAreaInput1ShapeSize = 4;
constexpr size_t kResizeAreaInput2ShapeSize = 1;
constexpr size_t kResizeAreaInputOutputdim1 = 0;
constexpr size_t kResizeAreaInputOutputdim2 = 1;
constexpr size_t kResizeAreaInputOutputdim3 = 2;
constexpr size_t kResizeAreaInputOutputdim4 = 3;
constexpr size_t kInput2ElementNum = 2;
int64_t ResizeAreaBound(int64_t val, int64_t limit) { return std::min(limit - 1, std::max(int64_t{0}, val)); }
}  // namespace

void ResizeAreaCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  input0_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input1_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  align_corners_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  if (input0_shape_.size() != kResizeAreaInput1ShapeSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the Images dimension should be "
                             << kResizeAreaInput1ShapeSize << ", but got " << input0_shape_.size();
  }
  if (input1_shape_.size() != kResizeAreaInput2ShapeSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the size dimension should be "
                             << kResizeAreaInput2ShapeSize << ", but got " << input1_shape_.size();
  }
  batch_size_ = input0_shape_[kResizeAreaInputOutputdim1];
  channels_ = input0_shape_[kResizeAreaInputOutputdim4];
  in_height_ = input0_shape_[kResizeAreaInputOutputdim2];
  in_width_ = input0_shape_[kResizeAreaInputOutputdim3];
}

bool ResizeAreaCPUKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeAreaInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeAreaOutputsNum, kernel_name_);
  auto out_size = reinterpret_cast<int32_t *>(inputs[1]->addr);
  if (inputs[1]->size / sizeof(size_type_) != kInput2ElementNum) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the num of elements in size should be 2, but got "
                             << inputs[1]->size / sizeof(size_type_);
  }
  if (out_size[0] <= 0 || out_size[1] <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the size must be positive, but got " << out_size[0]
                             << " and " << out_size[1];
  }
  out_height_ = out_size[0];
  out_width_ = out_size[1];
  height_scale_ = Scaling(in_height_, out_height_, align_corners_);
  width_scale_ = Scaling(in_width_, out_width_, align_corners_);
  std::vector<ResizeAreaCachedInterpolation> x_interps(out_width_);
  for (int32_t x = 0; x < out_width_; x++) {
    float transit_x0 = x * width_scale_;
    float transit_x1 = (x + 1) * width_scale_;
    size_t v = std::floor(transit_x0);
    x_interps[x].start = v;
    x_interps[x].start_scale = (v + 1 > transit_x1 ? width_scale_ : v + 1 - transit_x0);
    v = std::ceil(transit_x1);
    x_interps[x].end = v;
    v = x_interps[x].end - 1;
    x_interps[x].end_minus_one_scale = (v + 1 > transit_x1 ? transit_x1 - v : 1.0);
  }
  if (dtype_ == kNumberTypeFloat16) {
    return LaunchKernel<float16>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeFloat64) {
    return LaunchKernel<double>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeInt8) {
    return LaunchKernel<int8_t>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeInt16) {
    return LaunchKernel<int16_t>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeInt32) {
    return LaunchKernel<int32_t>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeInt64) {
    return LaunchKernel<int64_t>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeUInt8) {
    return LaunchKernel<uint8_t>(inputs, outputs, x_interps);
  } else if (dtype_ == kNumberTypeUInt16) {
    return LaunchKernel<uint16_t>(inputs, outputs, x_interps);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', unsupported input data type: " << dtype_;
    return false;
  }
  return true;
}

template <typename T>
bool ResizeAreaCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                                          const std::vector<ResizeAreaCachedInterpolation> &x_interps) const {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  float scale = 1.0 / (height_scale_ * width_scale_);
  auto out_size = reinterpret_cast<int32_t *>(inputs[1]->addr);
  std::vector<float> y_scales;
  std::vector<const T *> y_ptrs;
  for (int64_t b = 0; b < batch_size_; ++b) {
    for (int32_t y = 0; y < out_height_; ++y) {
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
      for (int32_t x = 0; x < out_width_; ++x) {
        const ResizeAreaCachedInterpolation &x_interp = x_interps[x];
        if (x_interp.needs_bounding) {
          ComputePatchSum<true>(scale, y_ptrs, y_scales, x_interp, output_addr);
        } else {
          ComputePatchSum<false>(scale, y_ptrs, y_scales, x_interp, output_addr);
        }
        output_addr += channels_;
      }
    }
  }
  SetResizeAreaOutShape(out_size);
  return true;
}

void ResizeAreaCPUKernelMod::SetResizeAreaOutShape(int32_t *out_size) const {
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < kResizeAreaInput1ShapeSize; i++) {
    if (i == kResizeAreaInputOutputdim1 || i == kResizeAreaInputOutputdim4) {
      out_shape.push_back(input0_shape_[i]);
    }
    if (i == kResizeAreaInputOutputdim2 || i == kResizeAreaInputOutputdim3) {
      out_shape.push_back(out_size[i - 1]);
    }
  }
  std::vector<TypeId> dtypes(1);
  dtypes[0] = AnfAlgo::GetOutputDeviceDataType(cnode_ptr_, 0);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape}, cnode_ptr_.get());
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
      float sum_y = static_cast<float>(ptr[channels_ * BOUND_IF_NEEDED(x_interp.start, in_width_) + c]) * scale_x;
      if (x_interp.start + 1 != x_interp.end) {
        for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
          sum_y += static_cast<float>(ptr[channels_ * BOUND_IF_NEEDED(x, in_width_) + c]);
        }
        scale_x = x_interp.end_minus_one_scale;
        sum_y += static_cast<float>(ptr[channels_ * BOUND_IF_NEEDED(x_interp.end - 1, in_width_) + c]) * scale_x;
      }
      sum += sum_y * y_scales[i];
    }
    output_patch_ptr[c] = sum * scale;
  }
#undef BOUND_IF_NEEDED
}

std::vector<KernelAttr> ResizeAreaCPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeArea, ResizeAreaCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
