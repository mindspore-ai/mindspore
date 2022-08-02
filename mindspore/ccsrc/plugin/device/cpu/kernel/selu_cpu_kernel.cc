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

#include "plugin/device/cpu/kernel/selu_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <map>
#include <functional>

namespace mindspore {
namespace kernel {
bool SeluCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int SeluCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  output_shape_.clear();
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToSize);
  output_size_ = std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies<size_t>());
  return KRET_OK;
}

template <typename T>
bool SeluCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  // The below alpha value and scale value is predefined, according to https://arxiv.org/abs/1706.02515
  T scale = static_cast<T>(1.05070098);
  T scale_dot_alpha = static_cast<T>(1.67326324 * 1.05070098);
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto task = [&input, &output, &scale_dot_alpha, &scale](size_t start, size_t end) {
    T template_zero = static_cast<T>(0);
    for (size_t i = start; i < end; i++) {
      T input_value = input[i];
      output[i] = input_value >= template_zero
                    ? (scale * input_value)
                    : (scale_dot_alpha * static_cast<T>(std::expm1(static_cast<float>(input_value))));
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_, pool_);
  return true;
}

const std::vector<std::pair<KernelAttr, SeluCpuKernelMod::KernelRunFunc>> &SeluCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SeluCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &SeluCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &SeluCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SeluCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SeluCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SeluCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SeLU, SeluCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
