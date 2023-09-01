/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/memcpy_cpu_kernel.h"
#include "mindspore/core/ops/array_ops.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMemcpyOutputsNum = 1;
constexpr auto kReshape = "Reshape";
constexpr auto kFlatten = "Flatten";
constexpr auto kFlattenGrad = "FlattenGrad";
constexpr auto kExpandDims = "ExpandDims";
constexpr auto kSqueeze = "Squeeze";
}  // namespace

bool MemcpyCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &,
                              const std::vector<KernelTensorPtr> &) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  return true;
}

int MemcpyCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  auto shape0 = inputs[kIndex0]->GetShapeVector();
  is_empty_tensor_ = std::any_of(shape0.begin(), shape0.end(), [](const int64_t shape) { return shape == 0; });
  return ret;
}

bool MemcpyCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  if (is_empty_tensor_) {
    return true;
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the inputs can not be empty.";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMemcpyOutputsNum, kernel_name_);
  if (inputs[0]->size != outputs[0]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'input_x': {" << inputs[0]->size
                      << "} is not equal to the size of the first output: {" << outputs[0]->size << "}";
  }
  if (inputs[0]->addr == outputs[0]->addr) {
    return true;
  }
  const auto *input_addr = reinterpret_cast<unsigned char *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<unsigned char *>(outputs[0]->addr);
  int cp_ret = EOK;
  auto task = [input_addr, output_addr, &cp_ret](size_t start, size_t end) {
    // the max size allowed by memcpy_s is SECUREC_MEM_MAX_LEN. If end - start > SECUREC_MEM_MAX_LEN,
    // we need to do memcpy_s multiple times instead of just one copy.
    do {
      auto size = end - start <= SECUREC_MEM_MAX_LEN ? end - start : SECUREC_MEM_MAX_LEN;
      auto ret = memcpy_s(output_addr + start, size, input_addr + start, size);
      start += SECUREC_MEM_MAX_LEN;
      if (ret != EOK && cp_ret == EOK) {
        cp_ret = ret;
        break;
      }
    } while (start < end);
  };
  ParallelLaunchAutoSearch(task, outputs[0]->size, this, &parallel_search_info_);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
  }
  return true;
}

std::vector<KernelAttr> MemcpyCpuKernelMod::common_valid_types_with_bool_complex_ = {
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
  KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
  KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128)};

std::vector<KernelAttr> MemcpyCpuKernelMod::common_two_valid_types_with_bool_complex_ = {
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
  KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
  KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
  KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
  KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
  KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
  KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128)};

std::vector<KernelAttr> MemcpyCpuKernelMod::GetOpSupport() {
  static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
    {kReshape, common_two_valid_types_with_bool_complex_},     {kFlatten, common_valid_types_with_bool_complex_},
    {kFlattenGrad, common_two_valid_types_with_bool_complex_}, {kExpandDims, common_two_valid_types_with_bool_complex_},
    {kSqueeze, common_valid_types_with_bool_complex_},
  };

  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
  }
  return iter->second;
}
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Reshape,
                                 []() { return std::make_shared<MemcpyCpuKernelMod>(kReshape); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Flatten,
                                 []() { return std::make_shared<MemcpyCpuKernelMod>(kFlatten); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FlattenGrad,
                                 []() { return std::make_shared<MemcpyCpuKernelMod>(kFlattenGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ExpandDims,
                                 []() { return std::make_shared<MemcpyCpuKernelMod>(kExpandDims); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Squeeze,
                                 []() { return std::make_shared<MemcpyCpuKernelMod>(kSqueeze); });
}  // namespace kernel
}  // namespace mindspore
