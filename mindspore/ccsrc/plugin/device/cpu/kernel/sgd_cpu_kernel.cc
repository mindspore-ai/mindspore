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

#include "plugin/device/cpu/kernel/sgd_cpu_kernel.h"
#include <algorithm>
#include <thread>
#include <vector>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSGDInputsNum = 6;
constexpr size_t kSGDOutputsNum = 1;
}  // namespace
void SGDCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dampening_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "dampening");
  weight_decay_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "weight_decay");
  nesterov_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "nesterov");

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SGD does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool SGDCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSGDInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSGDOutputsNum, kernel_name_);
  auto param = reinterpret_cast<T *>(inputs[PARAM]->addr);
  auto grad = reinterpret_cast<T *>(inputs[GRAD]->addr);
  auto lr = reinterpret_cast<T *>(inputs[LR]->addr);
  auto accum = reinterpret_cast<T *>(inputs[ACCUM]->addr);
  auto momentum = reinterpret_cast<T *>(inputs[MOMENTUM]->addr);
  auto stat = reinterpret_cast<T *>(inputs[STAT]->addr);
  auto output_param = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(T);

  auto task = [this, &param, &grad, &lr, &accum, &momentum, &stat, &output_param](size_t start, size_t end) {
    T ZERO = static_cast<T>(0);
    T ONE = static_cast<T>(1);
    for (size_t i = start; i < end; i++) {
      T grad_new = grad[i];
      if (weight_decay_ > static_cast<float>(0.0)) {
        grad_new += param[i] * static_cast<T>(weight_decay_);
      }
      if (momentum[0] > ZERO) {
        if (stat[i] > ZERO) {
          accum[i] = grad_new;
          stat[i] = ZERO;
        } else {
          accum[i] = accum[i] * momentum[0] + (ONE - static_cast<T>(dampening_)) * grad_new;
        }
        if (nesterov_) {
          grad_new += accum[i] * momentum[0];
        } else {
          grad_new = accum[i];
        }
      }
      param[i] -= lr[0] * grad_new;
      output_param[i] = param[i];
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, SGDCpuKernelMod::SGDFunc>> SGDCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &SGDCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &SGDCpuKernelMod::LaunchKernel<float16>}};

std::vector<KernelAttr> SGDCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SGDFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SGD, SGDCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
