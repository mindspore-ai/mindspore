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

#include "plugin/device/cpu/kernel/unsorted_segment_arithmetic_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUnsortedSegmentArithInputsNum = 2;
constexpr size_t kUnsortedSegmentArithOutputsNum = 1;
}  // namespace
#define UNSORTED_SEGNMENT_ARITHMETIC_CPU_REGISTER(T_DT, S_DT, T, S)       \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddOutputAttr(T_DT), \
    &UnsortedSegmentArithmeticCpuKernelMod::LaunchKernel<T, S>

template <typename T, typename S>
bool UnsortedSegmentArithmeticCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  static const std::map<std::string, T> UnsortedSegmentArithmeticInitValueMap{
    {prim::kPrimUnsortedSegmentMax->name(), std::numeric_limits<T>::lowest()},
    {prim::kPrimUnsortedSegmentMin->name(), std::numeric_limits<T>::max()}};

  if (UnsortedSegmentArithmeticInitValueMap.find(kernel_name_) == UnsortedSegmentArithmeticInitValueMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the current operator does not support this operation.";
    return false;
  }
  T init_value = UnsortedSegmentArithmeticInitValueMap.at(kernel_name_);

  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  S *indices_addr = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  for (size_t i = 0; i < out_size_; i++) {
    output_addr[i] = init_value;
  }

  if (kernel_name_ == prim::kPrimUnsortedSegmentMax->name()) {
    for (size_t loop = 0; loop < loop_size_; loop++) {
      auto output_index = indices_addr[loop];
      T *cur_input = input_addr + loop * comp_size_;
      T *cur_output = output_addr + output_index * comp_size_;
      for (size_t comp = 0; comp < comp_size_; comp++) {
        cur_output[comp] = cur_input[comp] > cur_output[comp] ? cur_input[comp] : cur_output[comp];
      }
    }
  } else if (kernel_name_ == prim::kPrimUnsortedSegmentMin->name()) {
    for (size_t loop = 0; loop < loop_size_; loop++) {
      auto output_index = indices_addr[loop];
      T *cur_input = input_addr + loop * comp_size_;
      T *cur_output = output_addr + output_index * comp_size_;
      for (size_t comp = 0; comp < comp_size_; comp++) {
        cur_output[comp] = cur_input[comp] < cur_output[comp] ? cur_input[comp] : cur_output[comp];
      }
    }
  }
  return true;
}

bool UnsortedSegmentArithmeticCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &inputs,
                                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUnsortedSegmentArithInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUnsortedSegmentArithOutputsNum, kernel_name_);

  return true;
}

int UnsortedSegmentArithmeticCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                  const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs,
                                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto in_shape = inputs[kIndex0]->GetShapeVector();
  auto out_shape = outputs[kIndex0]->GetShapeVector();

  comp_size_ = 1;
  out_size_ = out_shape[0];
  for (size_t i = 1; i < out_shape.size(); i++) {
    comp_size_ *= out_shape[i];
    out_size_ *= out_shape[i];
  }
  loop_size_ = 1;
  for (size_t i = 0; i < in_shape.size(); i++) {
    loop_size_ *= in_shape[i];
  }
  loop_size_ /= comp_size_;
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, UnsortedSegmentArithmeticCpuKernelMod::KernelRunFunc>>
  &UnsortedSegmentArithmeticCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, UnsortedSegmentArithmeticCpuKernelMod::KernelRunFunc>> func_list = {
    {UNSORTED_SEGNMENT_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double, int32_t)},
    {UNSORTED_SEGNMENT_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {UNSORTED_SEGNMENT_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float, int32_t)},
    {UNSORTED_SEGNMENT_ARITHMETIC_CPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {UNSORTED_SEGNMENT_ARITHMETIC_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t)},
    {UNSORTED_SEGNMENT_ARITHMETIC_CPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UnsortedSegmentMin, UnsortedSegmentArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UnsortedSegmentMax, UnsortedSegmentArithmeticCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
