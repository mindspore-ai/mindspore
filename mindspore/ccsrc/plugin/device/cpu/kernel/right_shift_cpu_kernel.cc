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

#include "plugin/device/cpu/kernel/right_shift_cpu_kernel.h"
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kRightShiftInputsNum = 2;
const size_t kRightShiftOutputsNum = 1;
}  // namespace

bool RightShiftCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRightShiftInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRightShiftOutputsNum, kernel_name_);
  input_type_1_ = inputs.at(kIndex0)->GetDtype();
  input_type_2_ = inputs.at(kIndex1)->GetDtype();
  if (input_type_1_ != input_type_2_) {
    MS_LOG(EXCEPTION) << "input1 and input2 must have the same type.";
  }
  return true;
}

int RightShiftCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_1_ = inputs.at(kIndex0)->GetShapeVector();
  input_shape_2_ = inputs.at(kIndex1)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  return KRET_OK;
}

bool RightShiftCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> & /* workspace */,
                                    const std::vector<AddressPtr> &outputs) {
  if (input_type_1_ == kNumberTypeInt8) {
    return IntCompute<int8_t>(inputs, outputs);
  } else if (input_type_1_ == kNumberTypeInt16) {
    return IntCompute<int16_t>(inputs, outputs);
  } else if (input_type_1_ == kNumberTypeInt32) {
    return IntCompute<int32_t>(inputs, outputs);
  } else if (input_type_1_ == kNumberTypeInt64) {
    return IntCompute<int64_t>(inputs, outputs);
  } else if (input_type_1_ == kNumberTypeUInt8) {
    return UIntCompute<uint8_t>(inputs, outputs);
  } else if (input_type_1_ == kNumberTypeUInt16) {
    return UIntCompute<uint16_t>(inputs, outputs);
  } else if (input_type_1_ == kNumberTypeUInt32) {
    return UIntCompute<uint32_t>(inputs, outputs);
  } else if (input_type_1_ == kNumberTypeUInt64) {
    return UIntCompute<uint64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'x' must be int8, int16, int32, int64, uint8, uint16, uint32, uint64, "
                         "but got "
                      << TypeIdLabel(input_type_1_);
  }
}

template <typename T>
bool RightShiftCpuKernelMod::IntCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto *input1 = static_cast<T *>(inputs[0]->addr);
  const auto *input2 = static_cast<T *>(inputs[1]->addr);
  auto *output = static_cast<T *>(outputs[0]->addr);
  if (output_shape_.size() == 0) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }
  int64_t size_tmp = 1;
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    size_tmp *= output_shape_[i];
  }
  size_t output_size = LongToSize(size_tmp);
  BroadcastIterator base_iter(input_shape_1_, input_shape_2_, output_shape_);
  auto task = [&input1, &input2, &output, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      T y_val = (input2[iter.GetInputPosB()]);
      T bit_val = static_cast<T>(sizeof(T) * 8 - 1);
      T zero = static_cast<T>(0);
      if (y_val <= zero) {
        y_val = zero;
      } else if (y_val > bit_val) {
        y_val = bit_val;
      }
      output[i] = static_cast<T>(input1[iter.GetInputPosA()] >> y_val);
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool RightShiftCpuKernelMod::UIntCompute(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto *input1 = static_cast<T *>(inputs[0]->addr);
  const auto *input2 = static_cast<T *>(inputs[1]->addr);
  auto *output = static_cast<T *>(outputs[0]->addr);
  if (output_shape_.size() == 0) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }
  int64_t size_tmp = 1;
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    size_tmp *= output_shape_[i];
  }
  size_t output_size = LongToSize(size_tmp);
  BroadcastIterator base_iter(input_shape_1_, input_shape_2_, output_shape_);
  auto task = [&input1, &input2, &output, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      T y_val = (input2[iter.GetInputPosB()]);
      T bit_val = static_cast<T>(sizeof(T) * 8 - 1);
      if (y_val > bit_val) {
        y_val = bit_val;
      }
      output[i] = static_cast<T>(input1[iter.GetInputPosA()] >> y_val);
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

std::vector<KernelAttr> RightShiftCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RightShift, RightShiftCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
