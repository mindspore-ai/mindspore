/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/empty_like_cpu_kernel.h"
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace empty_like_cpu {
namespace {
constexpr size_t kEmptyLikeInputsNum = 4;
constexpr size_t kEmptyLikeOutputsNum = 1;
}  // namespace

bool EmptyLikeCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEmptyLikeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEmptyLikeOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
  }
  return true;
}

bool EmptyLikeCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &,
                                         const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEmptyLikeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEmptyLikeOutputsNum, kernel_name_);
  auto device_name_opt = inputs[kIndex2]->GetOptionalValueWithCheck<int64_t>();
  if (device_name_opt.has_value()) {
    auto device_name_enum = device_name_opt.value();
    if (device_name_enum != DEVICE_ASCEND && device_name_enum != DEVICE_NPU_LOWER) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the device must be 'Ascend' or 'npu' in GRAPH mode.";
    }
  }
  auto pin_memory_opt = inputs[kIndex3]->GetOptionalValueWithCheck<bool>();
  if (pin_memory_opt.has_value() && pin_memory_opt.value()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the pin_memory must be False in GRAPH mode.";
  }
  return true;
}

std::vector<KernelAttr> EmptyLikeCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  if (support_list.empty()) {
    size_t types_num = ops::common_mint_valid_type_ids_with_complex_and_bool_vec.size();
    for (size_t i = 0; i < types_num; i++) {
      for (size_t j = 0; j < types_num; j++) {
        support_list.push_back(KernelAttr()
                                 .AddInputAttr(ops::common_mint_valid_type_ids_with_complex_and_bool_vec[i])
                                 .AddOptionalInputAttr(kNumberTypeInt64)
                                 .AddOptionalInputAttr(kNumberTypeInt64)
                                 .AddOptionalInputAttr(kNumberTypeBool)
                                 .AddOutputAttr(ops::common_mint_valid_type_ids_with_complex_and_bool_vec[j]));
      }
    }
  }
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EmptyLike, EmptyLikeCpuKernelMod);
}  // namespace empty_like_cpu
}  // namespace kernel
}  // namespace mindspore
