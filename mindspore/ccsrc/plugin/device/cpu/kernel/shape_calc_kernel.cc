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

#include "plugin/device/cpu/kernel/shape_calc_kernel.h"
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool ShapeCalcCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ShapeCalc>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ShapeCalc ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  func_ = kernel_ptr->get_shape_func();
  outputs_ = outputs;
  return true;
}

int ShapeCalcCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    return ret;
  }
  outputs_ = outputs;
  is_need_retrieve_output_shape_ = (ret == KRET_UNKNOWN_OUT_SHAPE);
  inputs_size_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input_shape = inputs[i]->GetShapeVector();
    auto sz = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
    inputs_size_.push_back(sz);
  }
  return ret;
}

bool ShapeCalcCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  if (func_ == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', shape func pointer is nullptr";
    return false;
  }
  if (inputs.size() != inputs_size_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', inputs address list size must be equal to inputs shape list size, but got " << inputs.size()
                  << " vs " << inputs_size_.size();
    return false;
  }

  ShapeArray args;
  args.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input_addr = reinterpret_cast<int64_t *>(inputs[i]->addr);
    args.emplace_back(input_addr, input_addr + inputs_size_[i]);
  }
  outs_shape_ = func_(args);
  if (outputs.size() != outs_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', outputs address list size must be equal to the number of outputs of shape func, but got "
                  << outputs.size() << " vs " << outs_shape_.size();
    return false;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto output_addr = reinterpret_cast<int64_t *>(outputs[i]->addr);
    for (size_t j = 0; j < outs_shape_[i].size(); ++j) {
      output_addr[j] = outs_shape_[i][j];
    }
  }
  return true;
}

std::vector<KernelAttr> ShapeCalcCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)}};
  return support_list;
}

void ShapeCalcCpuKernelMod::SyncData() {
  for (size_t i = 0; i < outs_shape_.size(); ++i) {
    ShapeVector shape{static_cast<int64_t>(outs_shape_[i].size())};
    outputs_[i]->SetShapeVector(shape);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ShapeCalc, ShapeCalcCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
