/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/native/size_cpu_kernel.h"
#include <functional>
#include <map>
#include <type_traits>
#include <algorithm>
#include <tuple>
#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace size_cpu {
namespace {
const size_t kSizeInputsNum = 1;
const size_t kSizeOutputsNum = 1;
};  // namespace
bool SizeCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(tensor_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(ERROR) << "Can not match kernel based on given attr!";
    return false;
  }
  return true;
}

int SizeCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto shape_vector = inputs[kIndex0]->GetShapeVector();
  int64_t elements = 1;
  for (size_t i = 0; i < shape_vector.size(); i++) {
    elements *= shape_vector[i];
  }
  input_elements = elements;
  return KRET_OK;
}

bool SizeCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                              const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSizeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSizeOutputsNum, kernel_name_);
  auto output_data = reinterpret_cast<int64_t *>(outputs[kIndex0]->device_ptr());
  MS_EXCEPTION_IF_NULL(output_data);
  output_data[kIndex0] = input_elements;
  return true;
}

std::vector<KernelAttr> SizeCpuKernelMod::GetOpSupport() {
  return {
    KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt4).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeGLUInt).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
  };
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Size, SizeCpuKernelMod);
}  // namespace size_cpu
}  // namespace kernel
}  // namespace mindspore
