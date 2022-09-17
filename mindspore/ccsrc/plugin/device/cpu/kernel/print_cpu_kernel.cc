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

#include "plugin/device/cpu/kernel/print_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include <complex>
#include "ir/tensor.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
using mindspore::tensor::Tensor;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

bool PrintCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  for (size_t i = 0; i < inputs.size(); ++i) {
    TypeId type = inputs[i]->GetDtype();
    (void)data_types_.emplace_back(type);
  }
  return true;
}

int PrintCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  input_sizes_.clear();
  input_shapes_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    auto input_shape = inputs[i]->GetShapeVector();
    (void)input_shapes_.emplace_back(input_shape);
    int64_t size = input_shape.empty() ? 0 : 1;
    for (size_t j = 0; j < input_shape.size(); ++j) {
      size *= input_shape[j];
    }
    (void)input_sizes_.emplace_back(LongToSize(size));
  }
  return ret;
}

bool PrintCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    TypeId dtype = data_types_[i];
    auto iter = func_map_.find(dtype);
    if (iter == func_map_.end()) {
      std::vector<std::string> all_types;
      std::for_each(func_map_.begin(), func_map_.end(),
                    [&all_types](auto iter) { all_types.push_back(TypeIdToString(iter.first)); });
      MS_LOG(ERROR) << "Print supported data type are " << all_types << ", but got the '" << i
                    << "' input data type is " << TypeIdToString(dtype);
      return false;
    }
    kernel_func_ = iter->second;
    kernel_func_(this, i, inputs);
  }
  return true;
}

template <typename T>
void PrintCpuKernelMod::LaunchKernel(size_t index, const std::vector<kernel::AddressPtr> &inputs) {
  if (input_sizes_[index] == 0) {
    auto num = reinterpret_cast<T *>(inputs[index]->addr);
    if constexpr (std::is_same<T, char>::value) {
      std::cout << num << std::endl;
    } else {
      std::cout << *num << std::endl;
    }
  } else {
    Tensor tensor(data_types_[index], input_shapes_[index], inputs[index]->addr, input_sizes_[index] * sizeof(T));
    std::cout << tensor.ToStringNoLimit() << std::endl;
  }
}

std::map<TypeId, PrintCpuKernelMod::PrintFunc> PrintCpuKernelMod::func_map_ = {
  {kNumberTypeBool, &PrintCpuKernelMod::LaunchKernel<bool>},
  {kNumberTypeInt8, &PrintCpuKernelMod::LaunchKernel<int8_t>},
  {kNumberTypeInt16, &PrintCpuKernelMod::LaunchKernel<int16_t>},
  {kNumberTypeInt32, &PrintCpuKernelMod::LaunchKernel<int32_t>},
  {kNumberTypeInt64, &PrintCpuKernelMod::LaunchKernel<int64_t>},
  {kNumberTypeUInt8, &PrintCpuKernelMod::LaunchKernel<uint8_t>},
  {kNumberTypeUInt16, &PrintCpuKernelMod::LaunchKernel<uint16_t>},
  {kNumberTypeUInt32, &PrintCpuKernelMod::LaunchKernel<uint32_t>},
  {kNumberTypeUInt64, &PrintCpuKernelMod::LaunchKernel<uint64_t>},
  {kNumberTypeFloat16, &PrintCpuKernelMod::LaunchKernel<float16>},
  {kNumberTypeFloat32, &PrintCpuKernelMod::LaunchKernel<float>},
  {kNumberTypeFloat64, &PrintCpuKernelMod::LaunchKernel<double>},
  {kObjectTypeString, &PrintCpuKernelMod::LaunchKernel<char>},
  {kNumberTypeComplex64, &PrintCpuKernelMod::LaunchKernel<complex64>},
  {kNumberTypeComplex128, &PrintCpuKernelMod::LaunchKernel<complex128>},
};

std::vector<KernelAttr> PrintCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Print, PrintCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
