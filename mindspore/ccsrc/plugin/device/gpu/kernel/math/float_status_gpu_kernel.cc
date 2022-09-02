/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/float_status_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kFloatStatus = "FloatStatus";
constexpr auto kIsInf = "IsInf";
constexpr auto kIsNan = "IsNan";
constexpr auto kIsFinite = "IsFinite";
}  // namespace

bool FloatStatusGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "The kernel name must be in "
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, FloatStatusGpuKernelMod::FloatStatusOpFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return false;
  }

  size_t input_num = inputs.size();
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
  }
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;

  auto type_iter = kOpTypeMap.find(kernel_name_);
  kernel_type_ = type_iter->second;

  type_id_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
  return true;
}

int FloatStatusGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto shape = inputs.at(kIndex0)->GetShapeVector();

  input_size_ = type_id_size_ * SizeOf(shape);
  if (kernel_type_ == OP_STATUS) {
    output_size_ = sizeof(float);
  } else {
    output_size_ = input_size_ / type_id_size_ * sizeof(bool);
  }
  return KRET_OK;
}

bool FloatStatusGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = stream_ptr;
  return kernel_func_(this, inputs, outputs);
}

template <typename T>
bool FloatStatusGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *input = GetDeviceAddress<T>(inputs, 0);

  switch (kernel_type_) {
    case OP_STATUS: {
      float *output = GetDeviceAddress<float>(outputs, 0);
      FillDeviceArray(outputs[0]->size / sizeof(float), output, 0.0f, reinterpret_cast<cudaStream_t>(cuda_stream_));
      CalFloatStatus(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(cuda_stream_));
      break;
    }
    case OP_INF: {
      bool *output = GetDeviceAddress<bool>(outputs, 0);
      CalIsInf(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(cuda_stream_));
      break;
    }
    case OP_NAN: {
      bool *output = GetDeviceAddress<bool>(outputs, 0);
      CalIsNan(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(cuda_stream_));
      break;
    }
    case OP_FINITE: {
      bool *output = GetDeviceAddress<bool>(outputs, 0);
      CalIsFinite(input_size_ / sizeof(T), input, output, reinterpret_cast<cudaStream_t>(cuda_stream_));
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "FloatStatus type " << kernel_type_ << " is not supported.";
    }
  }
  return true;
}

std::vector<KernelAttr> FloatStatusGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "For 'FloatStatus op', the kernel name must be in "
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, FloatStatusGpuKernelMod::FloatStatusOpFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FloatStatusOpFunc> &item) { return item.first; });
  return support_list;
}

std::map<std::string, std::vector<std::pair<KernelAttr, FloatStatusGpuKernelMod::FloatStatusOpFunc>>>
  FloatStatusGpuKernelMod::kernel_attr_map_ = {
    {kFloatStatus,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &FloatStatusGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
       &FloatStatusGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
       &FloatStatusGpuKernelMod::LaunchKernel<double>}}},
    {kIsInf,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<double>}}},
    {kIsNan,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<double>}}},
    {kIsFinite,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
       &FloatStatusGpuKernelMod::LaunchKernel<double>}}}};

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, FloatStatus,
                                 []() { return std::make_shared<FloatStatusGpuKernelMod>(kFloatStatus); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, IsInf,
                                 []() { return std::make_shared<FloatStatusGpuKernelMod>(kIsInf); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, IsNan,
                                 []() { return std::make_shared<FloatStatusGpuKernelMod>(kIsNan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, IsFinite,
                                 []() { return std::make_shared<FloatStatusGpuKernelMod>(kIsFinite); });
}  // namespace kernel
}  // namespace mindspore
