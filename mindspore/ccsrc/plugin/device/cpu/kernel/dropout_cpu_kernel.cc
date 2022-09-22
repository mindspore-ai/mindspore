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

#include "plugin/device/cpu/kernel/dropout_cpu_kernel.h"

#include <algorithm>
#include <random>
#include <map>
#include <functional>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/dropout.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDropoutInputsNum = 1;
constexpr size_t kDropoutOutputsNum = 2;
}  // namespace

using FuncVec = const std::vector<std::pair<KernelAttr, DropoutCpuKernelMod::KernelRunFunc>>;

bool DropoutCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Dropout>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Dropout ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kDropoutInputsNum || outputs.size() != kDropoutOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output tensor number must be " << kDropoutInputsNum
                  << " and " << kDropoutOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }
  keep_prob_ = kernel_ptr->get_keep_prob();
  if (keep_prob_ <= 0.0 || keep_prob_ > 1.0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the 'keep_prob' must be in (0.0, 1.0], but got " << keep_prob_;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int DropoutCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  for (const auto &d : input_shape_) {
    tensor_size_ *= LongToSize(d);
  }
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool DropoutCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDropoutInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDropoutOutputsNum, kernel_name_);

  const auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto mask_addr = reinterpret_cast<T *>(outputs[1]->addr);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dis(keep_prob_);
  T scale = static_cast<T>(1.f / keep_prob_);
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    mask_addr[i] = static_cast<T>(dis(gen));
    output_addr[i] = mask_addr[i] * input_addr[i] * scale;
  }
  return true;
}

FuncVec &DropoutCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, DropoutCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &DropoutCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &DropoutCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &DropoutCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Dropout, DropoutCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
