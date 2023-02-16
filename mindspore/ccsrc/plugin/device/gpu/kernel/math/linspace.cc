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

#include "plugin/device/gpu/kernel/math/linspace.h"
#include "mindspore/core/ops/lin_space.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLinSpaceInputsNum = 3;
constexpr size_t kLinSpaceOutputsNum = 1;
}  // namespace
bool LinSpaceGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::LinSpace>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast LinSpace ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLinSpaceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLinSpaceOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'LinSpace', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}
int LinSpaceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input1_shape = inputs[kIndex0]->GetShapeVector();
  auto input2_shape = inputs[kIndex1]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input1_shape, kernel_name_, "start") ||
                   CHECK_SHAPE_NULL(input2_shape, kernel_name_, "stop") ||
                   CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return static_cast<int>(KRET_OK);
  }
  value_count_ = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool LinSpaceGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);
  T *start_addr = GetDeviceAddress<T>(inputs, 0);
  T *stop_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  calLinSpace(start_addr, stop_addr, value_count_, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, LinSpaceGpuKernelMod::LinSpaceFunc>> LinSpaceGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &LinSpaceGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LinSpaceGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LinSpaceGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LinSpaceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LinSpaceFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LinSpace, LinSpaceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
