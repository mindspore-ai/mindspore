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

#include "include/curand.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/arrays/upper_bound_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr int64_t INPUT_DIMS = 2;
bool UpperBoundGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::UpperBound>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [uint16, int8, int16, "
                  << "int32, int64, float16, float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  unit_out_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  return true;
}

int UpperBoundGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> sorted_x_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> values_shape_ = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  sorted_x_elements_ = std::accumulate(sorted_x_shape_.begin(), sorted_x_shape_.end(), 1, std::multiplies<int64_t>());
  values_elements_ = std::accumulate(values_shape_.begin(), values_shape_.end(), 1, std::multiplies<int64_t>());
  if (sorted_x_elements_ == 0 || values_elements_ == 0) {
    is_null_input_ = true;
  }
  int64_t sorted_x_dims = sorted_x_shape_.size();
  if (sorted_x_dims != INPUT_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'sorted_x' should be 2-D, but got "
                  << sorted_x_dims << "-D.";
    return false;
  }
  int64_t values_dims = values_shape_.size();
  if (values_dims != INPUT_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'values' should be 2-D, but got " << values_dims
                  << "-D.";
    return false;
  }
  sorted_x_row_ = sorted_x_shape_[0];
  sorted_x_col_ = sorted_x_shape_[1];
  values_row_ = values_shape_[0];
  values_col_ = values_shape_[1];
  if (sorted_x_row_ != values_row_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rows of 'sorted_x' and 'values' should be the same.";
    return false;
  }

  size_t sorted_x_size = sorted_x_elements_ * unit_size_;
  size_t values_size = values_elements_ * unit_size_;
  size_t output_size = values_elements_ * unit_out_size_;
  input_size_list_.emplace_back(sorted_x_size);
  input_size_list_.emplace_back(values_size);
  output_size_list_.emplace_back(output_size);

  return KRET_OK;
}

template <typename T, typename S>
bool UpperBoundGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  T *sorted_x = GetDeviceAddress<T>(inputs, 0);
  T *values = GetDeviceAddress<T>(inputs, 1);
  S *output = GetDeviceAddress<S>(outputs, 0);
  CalUpperBound(values_elements_, sorted_x_col_, values_col_, sorted_x, values, output, device_id_,
                reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, UpperBoundGpuKernelMod::UpperBoundFunc>> UpperBoundGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<int8_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<int16_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<int32_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<int64_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<uint16_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<half, int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &UpperBoundGpuKernelMod::LaunchKernel<double, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &UpperBoundGpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> UpperBoundGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpperBoundFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpperBound, UpperBoundGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
