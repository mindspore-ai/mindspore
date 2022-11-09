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

#include "plugin/device/cpu/kernel/nan_to_num_cpu_kernel.h"
#include "mindspore/core/ops/nan_to_num.h"
#include "mindspore/core/base/float16.h"

using std::isinf;
using std::isnan;

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNanToNumInputsNum = 1;
constexpr size_t kNanToNumOutputsNum = 1;
}  // namespace

bool NanToNumCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NanToNum>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast NanToNum ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  nan_value_ = kernel_ptr->get_nan_value();
  posinf_value_ = kernel_ptr->get_posinf_value();
  neginf_value_ = kernel_ptr->get_neginf_value();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int NanToNumCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  return 0;
}

template <typename T>
bool NanToNumCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNanToNumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNanToNumOutputsNum, kernel_name_);
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);

  T posinf_value = static_cast<T>(posinf_value_);
  T neginf_value = static_cast<T>(neginf_value_);
  T nan_value = static_cast<T>(nan_value_);
  size_t total = inputs[0]->size / sizeof(T);
  auto task = [&input, &output, &posinf_value, &neginf_value, &nan_value](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if (input[i] > static_cast<T>(0) && isinf(input[i])) {
        output[i] = posinf_value;
      } else if (input[i] < static_cast<T>(0) && isinf(input[i])) {
        output[i] = neginf_value;
      } else if (isnan(input[i])) {
        output[i] = nan_value;
      } else {
        output[i] = input[i];
      }
    }
  };
  ParallelLaunchAutoSearch(task, total, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, NanToNumCpuKernelMod::KernelRunFunc>> &NanToNumCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, NanToNumCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &NanToNumCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &NanToNumCpuKernelMod::LaunchKernel<float>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NanToNum, NanToNumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
