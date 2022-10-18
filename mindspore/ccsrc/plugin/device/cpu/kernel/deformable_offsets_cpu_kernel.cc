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

#include "plugin/device/cpu/kernel/deformable_offsets_cpu_kernel.h"
#include <memory>
#include "ops/deformable_offsets.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsSize = 2;
constexpr size_t kOutputsSize = 1;
constexpr size_t kStridesSize = 4;
constexpr size_t kPadsSize = 4;
constexpr size_t kKernelSizeSize = 2;
constexpr size_t kKernelSizeHIndex = 0;
constexpr size_t kKernelSizeWIndex = 1;
constexpr size_t kDilationsSize = 4;
constexpr size_t kXShapeSize = 4;
constexpr size_t kOutputShapeSize = 4;
constexpr size_t kPadTopIndex = 0;
constexpr size_t kPadLeftIndex = 2;
constexpr size_t kOffsetsSize = 3;

template <typename T>
T DeformableBilinear(const T *input, T x, T y, int64_t width, int64_t height) {
  if (y <= static_cast<T>(-1) || y >= static_cast<T>(height) || x <= static_cast<T>(-1) || x >= static_cast<T>(width)) {
    return static_cast<T>(0);
  }
  int64_t left;
  if constexpr (std::is_same<T, float>::value) {
    left = static_cast<int64_t>(floorf(x));
  } else {
    left = static_cast<int64_t>(floor(x));
  }
  auto right = left + 1;
  int64_t top;
  if constexpr (std::is_same<T, float>::value) {
    top = static_cast<int64_t>(floorf(y));
  } else {
    top = static_cast<int64_t>(floor(y));
  }
  auto bottom = top + 1;

  T l = x - static_cast<T>(left);
  T r = static_cast<T>(1) - l;
  T t = y - static_cast<T>(top);
  T b = static_cast<T>(1) - t;

  T lt = static_cast<T>(0);
  T lb = static_cast<T>(0);
  if (left >= 0) {
    if (top >= 0) {
      lt = input[top * width + left];
    }
    if (bottom <= height - 1) {
      lb = input[bottom * width + left];
    }
  }
  T rt = static_cast<T>(0);
  T rb = static_cast<T>(0);
  if (right <= width - 1) {
    if (top >= 0) {
      rt = input[top * width + right];
    }
    if (bottom <= height - 1) {
      rb = input[bottom * width + right];
    }
  }

  T w_lt = r * b;
  T w_rt = l * b;
  T w_lb = r * t;
  T w_rb = l * t;
  T val = (w_lt * lt + w_rt * rt + w_lb * lb + w_rb * rb);
  return val;
}
}  // namespace

bool DeformableOffsetsCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputsSize || outputs.size() != kOutputsSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it should get " << kInputsSize << " inputs and " << kOutputsSize
                  << " outputs, but got " << inputs.size() << " inputs and " << outputs.size() << " outputs.";
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::DeformableOffsets>(base_operator->GetPrim());
  // Check args.
  n_axis_ = kIndex0;
  c_axis_ = kIndex1;
  h_axis_ = kIndex2;
  w_axis_ = kIndex3;
  strides_ = kernel_ptr->get_strides();
  if (strides_.size() != kStridesSize || strides_[n_axis_] != 1 || strides_[c_axis_] != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'strides' should be a vector with size " << kStridesSize
                  << " and the values according to N and C dimensions must be set to 1. But got 'strides': "
                  << strides_;
    return false;
  }
  pads_ = kernel_ptr->get_pads();
  if (pads_.size() != kPadsSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'pads' should be a vector with size " << kPadsSize
                  << ". But got 'pads': " << pads_;
    return false;
  }
  kernel_size_ = kernel_ptr->get_kernel_size();
  if (kernel_size_.size() != kKernelSizeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'kernel_size' should be a vector with size " << kKernelSizeSize
                  << ". But got 'kernel_size': " << kernel_size_;
    return false;
  }
  dilations_ = kernel_ptr->get_dilations();
  if (dilations_.size() != kDilationsSize || dilations_[n_axis_] != 1 || dilations_[c_axis_] != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dilations' should be a vector with size " << kDilationsSize
                  << " and the values according to N and C dimensions must be set to 1. But got 'dilations': "
                  << dilations_;
    return false;
  }
  deformable_groups_ = kernel_ptr->get_deformable_groups();
  if (deformable_groups_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'deformable_groups' should be greater than 0, but got "
                  << deformable_groups_;
    return false;
  }
  modulated_ = kernel_ptr->get_modulated();
  if (!modulated_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'modulated' only support to be set to True.";
    return false;
  }

  return MatchKernelFunc(base_operator, inputs, outputs);
}

void DeformableOffsetsCpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

int DeformableOffsetsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto x_shape = inputs.at(kIndex0)->GetShapeVector();
  if (x_shape.size() != kXShapeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape size of input 'x' should be " << kXShapeSize
                  << ", but got " << x_shape.size();
  }
  n_ = x_shape[n_axis_];
  c_ = x_shape[c_axis_];
  input_h_ = x_shape[h_axis_];
  input_w_ = x_shape[w_axis_];

  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  if (output_shape.size() != kOutputShapeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape size of output 'y' should be " << kOutputShapeSize
                  << ", but got " << output_shape.size();
  }
  output_h_ = output_shape[h_axis_];
  output_w_ = output_shape[w_axis_];
  position_grid_size_ = output_h_ * output_w_;
  (void)workspace_size_list_.emplace_back(sizeof(int64_t) * LongToSize(position_grid_size_) * kKernelSizeSize);
  return KRET_OK;
}

void DeformableOffsetsCpuKernelMod::GenPositionGrid(int64_t *position_grid) {
  auto task = [this, &position_grid](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto long_i = SizeToLong(i);
      int64_t y = long_i / output_w_;
      int64_t x = long_i % output_w_;
      int64_t pixel_y = y / kernel_size_[kKernelSizeHIndex];
      int64_t pixel_x = x / kernel_size_[kKernelSizeWIndex];
      int64_t kernel_y = y % kernel_size_[kKernelSizeHIndex];
      int64_t kernel_x = x % kernel_size_[kKernelSizeWIndex];
      size_t index = i * 2;
      position_grid[index] = pixel_x * strides_[w_axis_] + kernel_x * dilations_[w_axis_] - pads_[kPadLeftIndex];
      position_grid[index + 1] = pixel_y * strides_[h_axis_] + kernel_y * dilations_[h_axis_] - pads_[kPadTopIndex];
    }
  };
  ParallelLaunchAutoSearch(task, LongToSize(output_h_ * output_w_), this, &parallel_search_info_);
}

template <typename T>
void DeformableOffsetsCpuKernelMod::DeformableOffsets(const T *input_addr, const T *offsets_addr,
                                                      const int64_t *position_grid_addr, T *output_addr) {
  int64_t pixel_h = output_h_ / kernel_size_[kKernelSizeHIndex];
  int64_t pixel_w = output_w_ / kernel_size_[kKernelSizeWIndex];
  int64_t output_c_dim = output_h_ * output_w_;
  int64_t output_n_dim = c_ * output_c_dim;
  int64_t c_size_per_dfm_group = c_ / deformable_groups_;
  int64_t offset_kw_dim = pixel_h * pixel_w;
  int64_t offset_kh_dim = offset_kw_dim * kernel_size_[kKernelSizeWIndex];
  int64_t offset_group_dim = offset_kh_dim * kernel_size_[kKernelSizeHIndex];
  int64_t offset_mask_dim = offset_group_dim * deformable_groups_;
  int64_t offset_n_dim = offset_mask_dim * SizeToLong(kOffsetsSize);
  int64_t input_c_dim = input_h_ * input_w_;
  int64_t input_n_dim = input_c_dim * c_;

  auto task = [this, &input_addr, &offsets_addr, &output_addr, &position_grid_addr, &pixel_w, &output_c_dim,
               &output_n_dim, &c_size_per_dfm_group, &offset_kw_dim, &offset_kh_dim, &offset_group_dim,
               &offset_mask_dim, &offset_n_dim, &input_c_dim, &input_n_dim](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto long_i = SizeToLong(i);
      // Get input position
      int64_t hw_idx = long_i % output_c_dim;
      int64_t position_grid_idx = hw_idx * 2;
      int64_t input_x = position_grid_addr[position_grid_idx];
      int64_t input_y = position_grid_addr[position_grid_idx + 1];
      // Get offsets
      int64_t n_index = long_i / output_n_dim;
      int64_t c_index = long_i / output_c_dim % c_;
      int64_t x = hw_idx % output_w_;
      int64_t y = hw_idx / output_w_;
      int64_t dfm_group_index = c_index / c_size_per_dfm_group;
      int64_t pixel_x = x / kernel_size_[kKernelSizeWIndex];
      int64_t pixel_y = y / kernel_size_[kKernelSizeHIndex];
      int64_t kernel_x = x % kernel_size_[kKernelSizeWIndex];
      int64_t kernel_y = y % kernel_size_[kKernelSizeHIndex];
      int64_t x_offsets_offset = n_index * offset_n_dim + dfm_group_index * offset_group_dim +
                                 kernel_y * offset_kh_dim + kernel_x * offset_kw_dim + pixel_y * pixel_w + pixel_x;
      T x_offsets = offsets_addr[x_offsets_offset];
      int64_t y_offsets_offset = x_offsets_offset + offset_mask_dim;
      T y_offsets = offsets_addr[y_offsets_offset];
      int64_t mask_offset = y_offsets_offset + offset_mask_dim;
      T mask = offsets_addr[mask_offset];

      T new_x = static_cast<T>(input_x) + x_offsets;
      T new_y = static_cast<T>(input_y) + y_offsets;
      const T *input_addr_offset = input_addr + n_index * input_n_dim + c_index * input_c_dim;
      T bilinear_val = DeformableBilinear(input_addr_offset, new_x, new_y, input_w_, input_h_);
      output_addr[i] = bilinear_val * mask;
    }
  };
  ParallelLaunchAutoSearch(task, LongToSize(n_ * output_n_dim), this, &parallel_search_info_);
}

template <typename T>
bool DeformableOffsetsCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspaces,
                                                 const std::vector<AddressPtr> &outputs) {
  auto *position_grid_addr = GetDeviceAddress<int64_t>(workspaces, kIndex0);
  GenPositionGrid(position_grid_addr);
  T *x_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *offsets_addr = GetDeviceAddress<T>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  DeformableOffsets(x_addr, offsets_addr, position_grid_addr, output_addr);
  return true;
}

using KernelAttrAndDeformableOffsetsFuncList =
  std::vector<std::pair<KernelAttr, DeformableOffsetsCpuKernelMod::KernelRunFunc>>;
const KernelAttrAndDeformableOffsetsFuncList &DeformableOffsetsCpuKernelMod::GetFuncList() const {
  static const KernelAttrAndDeformableOffsetsFuncList func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &DeformableOffsetsCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &DeformableOffsetsCpuKernelMod::LaunchKernel<float>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DeformableOffsets, DeformableOffsetsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
