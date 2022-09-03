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

#include "plugin/device/cpu/kernel/convert_to_dynamic_cpu_kernel.h"
#include <functional>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kConvertToDynamicRank = "ConvertToDynamic";
}  // namespace
bool ConvertToDynamicCpuKernelMod::Init(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &,
                                        const std::vector<KernelTensorPtr> &) {
  kernel_name_ = kConvertToDynamicRank;
  is_need_retrieve_output_shape_ = true;
  return true;
}

int ConvertToDynamicCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KRET_OK;
  workspace_size_list_.clear();
  input_size_list_.clear();
  output_size_list_.clear();

  const size_t kInOutNum = 1;
  if (inputs.size() != kInOutNum || outputs.size() != kInOutNum) {
    MS_LOG(ERROR) << "Inputs or outputs size is requires 1, but got " << inputs.size() << " and " << outputs.size();
    return static_cast<int>(ret);
  }
  auto &input = inputs[0];
  size_t tensor_size = 0;
  size_t type_size = GetTypeByte(TypeIdToType(input->GetDtype()));
  auto shape = input->GetShapeVector();
  if (!IsValidShape(shape)) {
    ret = KRET_UNKNOWN_SHAPE;
    tensor_size = type_size;
  } else {
    tensor_size =
      shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    tensor_size = std::max(tensor_size, type_size);
  }

  (void)input_size_list_.emplace_back(tensor_size);
  (void)output_size_list_.emplace_back(tensor_size);

  outputs_ = inputs;
  return static_cast<int>(ret);
}

bool ConvertToDynamicCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "For 'ConvertToDynamic', the inputs can not be empty.";
  }
  const size_t kOutputsNum = 1;
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (inputs[0]->size != outputs[0]->size) {
    MS_LOG(EXCEPTION) << "For 'ConvertToDynamic', the size of 'input_x': {" << inputs[0]->size
                      << "} is not equal to the size of the first output: {" << outputs[0]->size << "}";
  }
  if (inputs[0]->addr == outputs[0]->addr) {
    return true;
  }
  size_t copy_size = outputs[0]->size;
  auto ret = memcpy_s(outputs[0]->addr, copy_size, inputs[0]->addr, copy_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For 'ConvertToDynamic', memcpy_s error. Error no: " << ret;
  }
  return true;
}

std::vector<KernelAttr> ConvertToDynamicCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> kernel_attr = {
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
    KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
    // double input
    // index int64
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
    // index int32
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
    // complex
    KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeComplex128),
    KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeComplex128),
  };
  return kernel_attr;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ConvertToDynamic, ConvertToDynamicCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
