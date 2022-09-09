/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/masked_select_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <map>
#include <complex>
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaskedSelectInputsNum = 2;
constexpr size_t kMaskedSelectOutputsNum = 1;
}  // namespace

bool MaskedSelectCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedSelectInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedSelectOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  is_need_retrieve_output_shape_ = true;  // MaskedSelect is a dynamic shape operator.
  kernel_func_ = func_list_[index].second;
  return true;
}

void MaskedSelectCpuKernelMod::ResetResource() noexcept {
  real_output_size_ = 0;
  tensor_size_ = 0;
  input_shape_a_.clear();
  input_shape_b_.clear();
  output_shape_.clear();
}

int MaskedSelectCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, others);
  if (ret != KRET_OK && ret != KRET_UNKNOWN_OUT_SHAPE) {
    return ret;
  }
  outputs_ = outputs;
  input_shape_a_ = inputs[kIndex0]->GetShapeVector();
  input_shape_b_ = inputs[kIndex1]->GetShapeVector();
  output_shape_ = CPUKernelUtils::GetBroadcastShape(input_shape_a_, input_shape_b_);
  tensor_size_ = LongToSize(std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies{}));
  return KRET_OK;
}

template <typename T>
bool MaskedSelectCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedSelectInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedSelectOutputsNum, kernel_name_);
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto mask = reinterpret_cast<bool *>(inputs[1]->addr);
  auto y = reinterpret_cast<T *>(outputs[0]->addr);
  size_t j = 0;
  if (input_shape_a_ == input_shape_b_) {
    for (size_t i = 0; i < tensor_size_; ++i) {
      if (mask[i]) {
        y[j++] = x[i];
      }
    }
  } else {  // Broadcast
    BroadcastIterator iter(input_shape_a_, input_shape_b_, output_shape_);
    iter.SetPos(0);
    for (size_t i = 0; i < tensor_size_; ++i) {
      if (mask[iter.GetInputPosB()]) {
        y[j++] = x[iter.GetInputPosA()];
      }
      iter.GenNextPos();
    }
  }

  real_output_size_ = j;
  return true;
}

void MaskedSelectCpuKernelMod::SyncData() {
  std::vector<int64_t> new_output_shape = {SizeToLong(real_output_size_)};
  outputs_[kIndex0]->SetShapeVector(new_output_shape);
}

std::vector<std::pair<KernelAttr, MaskedSelectCpuKernelMod::MaskedSelectFunc>> MaskedSelectCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt8),
   &MaskedSelectCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt16),
   &MaskedSelectCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
   &MaskedSelectCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
   &MaskedSelectCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat16),
   &MaskedSelectCpuKernelMod::LaunchKernel<Eigen::half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),
   &MaskedSelectCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat64),
   &MaskedSelectCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8),
   &MaskedSelectCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt16),
   &MaskedSelectCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt32),
   &MaskedSelectCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt64),
   &MaskedSelectCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &MaskedSelectCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeComplex64),
   &MaskedSelectCpuKernelMod::LaunchKernel<std::complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeComplex128),
   &MaskedSelectCpuKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> MaskedSelectCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaskedSelectFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaskedSelect, MaskedSelectCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
