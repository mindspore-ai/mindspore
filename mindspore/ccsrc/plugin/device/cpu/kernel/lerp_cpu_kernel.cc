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

#include "plugin/device/cpu/kernel/lerp_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <map>

namespace mindspore {
namespace kernel {
bool LerpCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int LerpCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto start_shape = inputs.at(kIndex0)->GetShapeVector();
  auto end_shape = inputs.at(kIndex1)->GetShapeVector();
  auto weight_shape = inputs.at(kIndex2)->GetShapeVector();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(start_shape.begin(), start_shape.end(), std::back_inserter(start_shape_), LongToSize);
  (void)std::transform(end_shape.begin(), end_shape.end(), std::back_inserter(end_shape_), LongToSize);
  (void)std::transform(weight_shape.begin(), weight_shape.end(), std::back_inserter(weight_shape_), LongToSize);
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToSize);
  output_size_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());
  return KRET_OK;
}

void LerpCpuKernelMod::ResetResource() noexcept {
  output_size_ = 0;
  start_shape_.clear();
  end_shape_.clear();
  weight_shape_.clear();
  output_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool LerpCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (start_shape_ == end_shape_ && start_shape_ == weight_shape_) {
    auto *input_start = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
    auto *input_end = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
    auto *input_weight = reinterpret_cast<T *>(inputs.at(kIndex2)->addr);
    T *output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
    auto task = [&input_start, &input_end, &input_weight, &output](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        T start_value = input_start[i];
        T end_value = input_end[i];
        T weight_value = input_weight[i];
        output[i] = static_cast<T>(start_value + (end_value - start_value) * weight_value);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_, pool_);
  } else {
    MultipleBroadcastIterator multi_broadcast_iterator({start_shape_, end_shape_, weight_shape_}, output_shape_);
    auto *input_start = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
    auto *input_end = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
    auto *input_weight = reinterpret_cast<T *>(inputs.at(kIndex2)->addr);
    T *output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
    auto task = [&input_start, &input_end, &input_weight, &output, &multi_broadcast_iterator](size_t start,
                                                                                              size_t end) {
      auto iter = multi_broadcast_iterator;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        T start_value = input_start[iter.GetInputPos(kIndex0)];
        T end_value = input_end[iter.GetInputPos(kIndex1)];
        T weight_value = input_weight[iter.GetInputPos(kIndex2)];
        output[i] = static_cast<T>(start_value + (end_value - start_value) * weight_value);
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_, pool_);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, LerpCpuKernelMod::KernelRunFunc>> &LerpCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, LerpCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &LerpCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &LerpCpuKernelMod::LaunchKernel<float>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Lerp, LerpCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
