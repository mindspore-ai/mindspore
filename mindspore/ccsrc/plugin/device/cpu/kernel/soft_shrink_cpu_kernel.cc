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

#include "plugin/device/cpu/kernel/soft_shrink_cpu_kernel.h"
#include "mindspore/core/ops/soft_shrink.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace kernel {
#define SOFT_SHRINK_CPU_REGISTER(DT, T) \
  KernelAttr().AddInputAttr(DT).AddOutputAttr(DT), &SoftShrinkCpuKernelMod::LaunchKernel<T>

template <typename T>
bool SoftShrinkCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  /* float optimize */
  if (std::is_same_v<T, float>) {
    float *input = reinterpret_cast<float *>(inputs.at(kIndex0)->addr);
    float *output = reinterpret_cast<float *>(outputs.at(kIndex0)->addr);

    auto task = [input, output, this](size_t start, size_t end) {
      auto input_tmp = input + start;
      auto output_tmp = output + start;
      (void)SoftShrink(input_tmp, (end - start), output_tmp, this->lambd_);
    };
    ParallelLaunchAutoSearch(task, size_, this, &parallel_search_info_);
    return true;
  }

  /* common soft shrink */
  T *input_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  T *output_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  T pos_lamdb = static_cast<T>(lambd_);
  T neg_lambd = static_cast<T>(-1 * pos_lamdb);
  auto task = [input_addr, output_addr, pos_lamdb, neg_lambd](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if (input_addr[i] > pos_lamdb) {
        output_addr[i] = input_addr[i] - pos_lamdb;
      } else if (input_addr[i] < neg_lambd) {
        output_addr[i] = input_addr[i] + pos_lamdb;
      } else {
        output_addr[i] = 0;
      }
    }
  };
  ParallelLaunchAutoSearch(task, size_, this, &parallel_search_info_);
  return true;
}

bool SoftShrinkCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::SoftShrink>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast SoftShrink ops failed!";
    return false;
  }
  lambd_ = kernel_ptr->get_lambd();

  if (auto ret = MatchKernelFunc(base_operator, inputs, outputs); !ret) {
    return ret;
  }
  return true;
}

int SoftShrinkCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  size_ = std::accumulate(in_shape.begin(), in_shape.end(), size_t(1), std::multiplies<size_t>());
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, SoftShrinkCpuKernelMod::KernelRunFunc>> &SoftShrinkCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, SoftShrinkCpuKernelMod::KernelRunFunc>> func_list = {
    {SOFT_SHRINK_CPU_REGISTER(kNumberTypeFloat32, float)},
    {SOFT_SHRINK_CPU_REGISTER(kNumberTypeInt32, int32_t)},
    {SOFT_SHRINK_CPU_REGISTER(kNumberTypeInt64, int64_t)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SoftShrink, SoftShrinkCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
