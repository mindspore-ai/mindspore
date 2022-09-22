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

#include "plugin/device/cpu/kernel/masked_select_grad_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <map>
#include <functional>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaskedSelectGradInputsNum = 3;
constexpr size_t kMaskedSelectGradOutputsNum = 1;
constexpr size_t kIndexInput = 0;
constexpr size_t kIndexMask = 1;
constexpr size_t kIndexGrad = 2;
constexpr size_t kIndexOutput = 0;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool MaskedSelectGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  input_shape_a_ = inputs[kIndexInput]->GetShapeVector();
  input_shape_b_ = inputs[kIndexMask]->GetShapeVector();
  grad_shape_ = inputs[kIndexGrad]->GetShapeVector();
  output_shape_ = CPUKernelUtils::GetBroadcastShape(input_shape_a_, input_shape_b_);
  tensor_size_ = 1;
  tensor_size_ =
    std::accumulate(output_shape_.cbegin(), output_shape_.cend(), tensor_size_, std::multiplies<int64_t>());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MaskedSelectGrad does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaskedSelectGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  input_shape_a_ = inputs[kIndexInput]->GetShapeVector();
  input_shape_b_ = inputs[kIndexMask]->GetShapeVector();
  grad_shape_ = inputs[kIndexGrad]->GetShapeVector();
  output_shape_ = CPUKernelUtils::GetBroadcastShape(input_shape_a_, input_shape_b_);
  const auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  tensor_size_ = 1;
  tensor_size_ =
    std::accumulate(output_shape_.cbegin(), output_shape_.cend(), tensor_size_, std::multiplies<int64_t>());
  return KRET_OK;
}

template <typename T>
bool MaskedSelectGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedSelectGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedSelectGradOutputsNum, kernel_name_);
  if (tensor_size_ < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', tensor_size_[" << std::to_string(tensor_size_)
                      << "] should not be less than zero. Output shape: " << output_shape_;
  }
  auto mask = reinterpret_cast<bool *>(inputs[kIndexMask]->addr);
  auto grad = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);
  auto dx = reinterpret_cast<T *>(outputs[kIndexInput]->addr);

  auto ret = memset_s(dx, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output[0] failed. Error no: " << ret;
  }

  uint64_t output_size = outputs[0]->size / sizeof(T);
  uint64_t j = 0;
  if (input_shape_a_ == input_shape_b_) {
    for (uint64_t i = 0; i < output_size; ++i) {
      if (mask[i]) {
        dx[i] += grad[j++];
      }
    }
  } else {
    BroadcastIterator iter(input_shape_a_, input_shape_b_, output_shape_);
    iter.SetPos(0);
    for (uint64_t i = 0; i < LongToUlong(tensor_size_); ++i) {
      if (mask[iter.GetInputPosB()]) {
        dx[iter.GetInputPosA()] += grad[j++];
      }
      iter.GenNextPos();
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, MaskedSelectGradCpuKernelMod::MaskedSelectGradFunc>>
  MaskedSelectGradCpuKernelMod::func_list_ = {{KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat16)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeFloat16)
                                                 .AddOutputAttr(kNumberTypeFloat16),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<float16>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat32)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeFloat32)
                                                 .AddOutputAttr(kNumberTypeFloat32),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<float>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat64)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeFloat64)
                                                 .AddOutputAttr(kNumberTypeFloat64),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<double>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt8)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeInt8)
                                                 .AddOutputAttr(kNumberTypeInt8),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<int8_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt16)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeInt16)
                                                 .AddOutputAttr(kNumberTypeInt16),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<int16_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt32),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt64)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeInt64)
                                                 .AddOutputAttr(kNumberTypeInt64),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<int64_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt8)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeUInt8)
                                                 .AddOutputAttr(kNumberTypeUInt8),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<uint8_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt16)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeUInt16)
                                                 .AddOutputAttr(kNumberTypeUInt16),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<uint16_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt32)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeUInt32)
                                                 .AddOutputAttr(kNumberTypeUInt32),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<uint32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt64)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeUInt64)
                                                 .AddOutputAttr(kNumberTypeUInt64),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<uint64_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddOutputAttr(kNumberTypeBool),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<int64_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeComplex64)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeComplex64)
                                                 .AddOutputAttr(kNumberTypeComplex64),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<complex64>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeComplex128)
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeComplex128)
                                                 .AddOutputAttr(kNumberTypeComplex128),
                                               &MaskedSelectGradCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> MaskedSelectGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaskedSelectGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaskedSelectGrad, MaskedSelectGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
