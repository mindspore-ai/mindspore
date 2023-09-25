/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
  auto operator_ptr = std::dynamic_pointer_cast<ops::ShapeCalc>(base_operator);
  if (!operator_ptr) {
    MS_LOG(ERROR) << "cast ShapeCalc ops failed!";
    return false;
  }
  kernel_name_ = operator_ptr->name();
  functor_ = operator_ptr->get_functor();
  value_depend_ = operator_ptr->get_value_depend();
  return true;
}

int ShapeCalcCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    return ret;
  }
  is_need_retrieve_output_shape_ = (ret == KRET_UNKNOWN_OUT_SHAPE);

  input_shapes_.clear();
  inputs_size_.clear();
  inputs_type_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input_shape = inputs[i]->GetShapeVector();
    input_shapes_.push_back(input_shape);

    auto sz = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
    inputs_size_.push_back(sz);

    auto type_id = inputs[i]->GetDtype();
    if (value_depend_[i] && type_id != kNumberTypeInt32 && type_id != kNumberTypeInt64) {
      MS_LOG(EXCEPTION) << "For ShapeCalc value-depended input[" << i << "] should be int32 or int64, but got "
                        << TypeIdToString(type_id);
    }
    inputs_type_.push_back(type_id);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto type_id = outputs[i]->GetDtype();
    if (type_id != kNumberTypeInt64) {
      MS_LOG(EXCEPTION) << "For ShapeCalc output should be int64, but got " << TypeIdToString(type_id);
    }
  }
  return ret;
}

ShapeVector ShapeCalcCpuKernelMod::GetInt64ValueFromAddr(void *addr, size_t idx) {
  if (inputs_type_.at(idx) == kNumberTypeInt32) {
    auto input_addr = reinterpret_cast<int32_t *>(addr);
    return ShapeVector{input_addr, input_addr + inputs_size_[idx]};
  }
  auto input_addr = reinterpret_cast<int64_t *>(addr);
  return ShapeVector{input_addr, input_addr + inputs_size_[idx]};
}

bool ShapeCalcCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  if (functor_ == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', shape func pointer is nullptr";
    return false;
  }
  if (inputs.size() != inputs_size_.size() || inputs.size() != inputs_type_.size()) {
    MS_LOG(ERROR)
      << "For '" << kernel_name_
      << "', inputs address list size must be equal to inputs shape list size and inputs dtype list size, but got "
      << inputs.size() << " vs " << inputs_size_.size() << " vs " << inputs_type_.size();
    return false;
  }

  ShapeArray args;
  args.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    args.push_back(value_depend_[i] ? GetInt64ValueFromAddr(inputs[i]->addr, i) : input_shapes_[i]);
  }

  outs_shape_ = functor_->Calc(args);
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
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true).AddRealTuple(true)};
  return support_list;
}

void ShapeCalcCpuKernelMod::SyncOutputShape() {
  for (size_t i = 0; i < outs_shape_.size(); ++i) {
    ShapeVector shape{static_cast<int64_t>(outs_shape_[i].size())};
    outputs_[i]->SetShapeVector(shape);
  }
}

std::vector<size_t> ShapeCalcCpuKernelMod::GetLaunchIgnoredInputAddressIdx() const {
  std::vector<size_t> ignored_idx;
  for (size_t i = 0; i < value_depend_.size(); ++i) {
    if (!value_depend_[i]) {
      ignored_idx.push_back(i);
    }
  }
  return ignored_idx;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ShapeCalc, ShapeCalcCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
