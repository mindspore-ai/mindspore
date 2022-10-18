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

#include "plugin/device/cpu/kernel/rgb_to_hsv_cpu_kernel.h"
#include <algorithm>
#include "Eigen/Core"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNumberOfRGB = 3;
const size_t kInputNum = 1;
const size_t kOutputNum = 1;
}  // namespace

bool RGBToHSVCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  input_dtype = inputs.at(kIndex0)->GetDtype();
  if (input_dtype != kNumberTypeFloat32 && input_dtype != kNumberTypeFloat64 && input_dtype != kNumberTypeFloat16) {
    MS_EXCEPTION(TypeError) << "For " << kernel_name_ << ", the type of inputs are invalid";
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RGBToHSVCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  input0_elements_nums_ = 1;
  for (size_t i = 0; i < input_shape.size(); i++) {
    input0_elements_nums_ *= static_cast<size_t>(input_shape[i]);
  }

  if (input_shape[input_shape.size() - 1] != static_cast<int64_t>(kNumberOfRGB)) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", the last dimension of the input tensor must be size 3.";
  }

  if (input_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", the dimension of the input tensor must be at least 1.";
  }

  return KRET_OK;
}

template <typename T>
bool RGBToHSVCpuKernelMod::ComputeFloat(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  T *input_data = reinterpret_cast<T *>(inputs[0]->addr);
  T *output_data = reinterpret_cast<T *>(outputs[0]->addr);
  for (size_t i = 0; i < input0_elements_nums_; i = i + kNumberOfRGB) {
    auto t_red = *(input_data + i);
    auto t_green = *(input_data + i + 1);
    auto t_blue = *(input_data + i + 2);
    auto t_value = std::max(std::max(t_red, t_blue), t_green);
    auto t_minimum = std::min(std::min(t_red, t_blue), t_green);
    auto range = t_value - t_minimum;
    T t_saturation = static_cast<T>(0);
    if (static_cast<T>(t_value) > static_cast<T>(0)) {
      t_saturation = static_cast<T>(range / static_cast<T>(t_value));
    }
    auto norm = static_cast<T>(1.0) / static_cast<T>(6.0) / range;
    auto t_hue = t_green == t_value ? (norm * (t_blue - t_red) + static_cast<T>(2.0) / static_cast<T>(6.0))
                                    : (norm * (t_red - t_green) + static_cast<T>(4.0) / static_cast<T>(6.0));
    t_hue = t_red == t_value ? (norm * (t_green - t_blue)) : t_hue;
    t_hue = range > static_cast<T>(0) ? t_hue : static_cast<T>(0);
    t_hue = t_hue < static_cast<T>(0) ? (t_hue + static_cast<T>(1)) : t_hue;
    *(output_data + i) = t_hue;
    *(output_data + i + 1) = t_saturation;
    *(output_data + i + 1 + 1) = t_value;
  }
  return true;
}

bool RGBToHSVCpuKernelMod::ComputeHalf(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  float16 *input_data = reinterpret_cast<float16 *>(inputs[0]->addr);
  float16 *output_data = reinterpret_cast<float16 *>(outputs[0]->addr);
  for (size_t i = 0; i < input0_elements_nums_; i = i + kNumberOfRGB) {
    auto t_red = *(input_data + i);
    auto t_green = *(input_data + i + 1);
    auto t_blue = *(input_data + i + 2);
    auto t_value = std::max(std::max(t_red, t_blue), t_green);
    auto t_minimum = std::min(std::min(t_red, t_blue), t_green);
    auto range = t_value - t_minimum;
    float t_saturation = static_cast<float>(0);
    if (static_cast<float>(t_value) > static_cast<float>(0)) {
      t_saturation = static_cast<float>(range / static_cast<float>(t_value));
    }
    auto norm = static_cast<float>(1.0) / static_cast<float>(6.0) / static_cast<float>(range);
    auto t_hue = t_green == t_value
                   ? (norm * static_cast<float>(t_blue - t_red) + static_cast<float>(2.0) / static_cast<float>(6.0))
                   : (norm * static_cast<float>(t_red - t_green) + static_cast<float>(4.0) / static_cast<float>(6.0));
    t_hue = t_red == t_value ? (norm * static_cast<float>(t_green - t_blue)) : t_hue;
    t_hue = static_cast<float>(range) > static_cast<float>(0) ? t_hue : static_cast<float>(0);
    t_hue = t_hue < static_cast<float>(0) ? (t_hue + static_cast<float>(1)) : t_hue;
    *(output_data + i) = float16(t_hue);
    *(output_data + i + 1) = float16(t_saturation);
    *(output_data + i + 1 + 1) = float16(t_value);
  }
  return true;
}

template <typename T>
bool RGBToHSVCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  TypeId dtype_{kTypeUnknown};
  dtype_ = input_dtype;

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);

  if (dtype_ == kNumberTypeFloat32) {
    res_ = ComputeFloat<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    res_ = ComputeFloat<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    res_ = ComputeHalf(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "For " << kernel_name_
                            << ", it does not support this input data type: " << TypeIdLabel(dtype_) << ".";
  }
  return res_;
}

std::vector<std::pair<KernelAttr, RGBToHSVCpuKernelMod::RGBToHSVFunc>> RGBToHSVCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &RGBToHSVCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &RGBToHSVCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &RGBToHSVCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> RGBToHSVCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RGBToHSVFunc> &pair) { return pair.first; });

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RGBToHSV, RGBToHSVCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
