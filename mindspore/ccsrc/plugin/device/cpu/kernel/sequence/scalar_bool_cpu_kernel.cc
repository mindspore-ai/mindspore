/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/scalar_bool_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

bool ScalarBoolCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For kernel '" << kernel_name_ << "' input_num must be 1, but got " << inputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ScalarBoolCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool ScalarBoolCpuKernelMod::LaunchKernel(const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::vector<AddressPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto input_x = reinterpret_cast<T *>(inputs[0]->GetData()->addr);
  auto output = reinterpret_cast<bool *>(outputs[0]->GetData()->addr);
  *output = static_cast<bool>(*input_x);
  return true;
}

#define ADD_KERNEL(in_dtype, out_dtype, in_type)                 \
  {                                                              \
    KernelAttr()                                                 \
      .AddInputAttr(kObjectTypeNumber, kNumberType##in_dtype)    \
      .AddOutputAttr(kObjectTypeNumber, kNumberType##out_dtype), \
      &ScalarBoolCpuKernelMod::LaunchKernel<in_type>             \
  }

std::vector<std::pair<KernelAttr, ScalarBoolCpuKernelMod::ScalarBoolFunc>> ScalarBoolCpuKernelMod::func_list_ = {
  ADD_KERNEL(Float32, Bool, float), ADD_KERNEL(Float64, Bool, double), ADD_KERNEL(Int32, Bool, int32_t),
  ADD_KERNEL(Int64, Bool, int64_t), ADD_KERNEL(Bool, Bool, bool)};

std::vector<KernelAttr> ScalarBoolCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ScalarBoolFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScalarBool, ScalarBoolCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
