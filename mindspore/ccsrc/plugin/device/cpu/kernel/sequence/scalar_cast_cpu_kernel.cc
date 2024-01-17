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

#include "plugin/device/cpu/kernel/sequence/scalar_cast_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool ScalarCastCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int ScalarCastCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto ele_shape = inputs[0]->GetShapeVector();
  if (!ele_shape.empty() && !(ele_shape.size() == 1 && ele_shape[0] == 1)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input shape should be 0 or 1, but got " << ele_shape;
  }
  return KRET_OK;
}

template <typename T>
bool ScalarCastCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &,
                                          const std::vector<KernelTensor *> &outputs) {
  const auto ele_addr = static_cast<T *>(inputs[kIndex0]->device_ptr());
  MS_EXCEPTION_IF_NULL(ele_addr);
  const auto input_type_addr = static_cast<TypeId *>(inputs[kIndex1]->device_ptr());
  MS_EXCEPTION_IF_NULL(input_type_addr);

  auto output_addr = outputs[kIndex0]->device_ptr();
  MS_EXCEPTION_IF_NULL(output_addr);
  if (*input_type_addr == TypeId::kNumberTypeInt64) {
    *(static_cast<int64_t *>(output_addr)) = static_cast<int64_t>(ele_addr[0]);
  } else if (*input_type_addr == TypeId::kNumberTypeFloat64) {
    *(static_cast<double *>(output_addr)) = static_cast<double>(ele_addr[0]);
  } else if (*input_type_addr == TypeId::kNumberTypeBool) {
    *(static_cast<bool *>(output_addr)) = static_cast<bool>(ele_addr[0]);
  } else {
    MS_LOG(EXCEPTION) << "For [ScalarCast], the output type should only be kNumberTypeInt64, kNumberTypeFloat64, or "
                         "kNumberTypeBool, but got: "
                      << *input_type_addr;
  }

  return true;
}

static const std::vector<std::pair<KernelAttr, ScalarCastCpuKernelMod::KernelRunFunc>> func_list = {
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
   &ScalarCastCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
   &ScalarCastCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
   &ScalarCastCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),
   &ScalarCastCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
   &ScalarCastCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
   &ScalarCastCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
   &ScalarCastCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),
   &ScalarCastCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
   &ScalarCastCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
   &ScalarCastCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
   &ScalarCastCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),
   &ScalarCastCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
   &ScalarCastCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
   &ScalarCastCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
   &ScalarCastCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),
   &ScalarCastCpuKernelMod::LaunchKernel<bool>}};

const std::vector<std::pair<KernelAttr, ScalarCastCpuKernelMod::KernelRunFunc>> &ScalarCastCpuKernelMod::GetFuncList()
  const {
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScalarCast, ScalarCastCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
