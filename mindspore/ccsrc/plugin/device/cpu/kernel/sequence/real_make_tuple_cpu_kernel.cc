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

#include "plugin/device/cpu/kernel/sequence/real_make_tuple_cpu_kernel.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <complex>
#include <vector>
#include "kernel/kernel.h"
#include "mindapi/base/type_id.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"
#include "base/user_data.h"
namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kOutputNum = 1;
}  // namespace

bool RealMakeTupleCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RealMakeTupleCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    const auto &shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      return static_cast<int>(KRET_UNKNOWN_SHAPE);
    }
  }

  workspace_size_list_.clear();
  output_size_list_.clear();

  if (inputs.empty() || outputs.empty()) {
    return static_cast<int>(KRET_RESIZE_FAILED);
  }

  std::vector<size_t> real_sizes;
  real_sizes.reserve(inputs.size());
  bool element_all_same = true;
  auto elem_size = inputs[0]->size();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto cur_input_size = inputs[i]->size();
    real_sizes.push_back(cur_input_size);
    if (elem_size < cur_input_size) {
      elem_size = cur_input_size;
      element_all_same = false;
    }
  }

  output_size_list_ = std::vector<size_t>{inputs.size() * elem_size};

  if (!element_all_same) {
    auto user_data = outputs[0]->user_data();
    if (user_data == nullptr) {
      user_data = std::make_shared<UserData>();
      outputs[0]->set_user_data(user_data);
    }
    user_data->set(kRealElementsSize, std::make_shared<std::vector<size_t>>(real_sizes));
  }

  return KRET_OK;
}

template <typename T>
bool RealMakeTupleCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &,
                                             const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  size_t elem_offset = (output_size_list_[0] / sizeof(T) / inputs.size());
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    T *input_addr = GetDeviceAddress<T>(inputs, i);
    auto input_size = inputs[i]->size();
    if (input_size != 0) {
      auto cp_ret = memcpy_s(output_addr, input_size, input_addr, input_size);
      if (cp_ret != EOK) {
        MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
      }
    }
    output_addr += elem_offset;
  }

  return true;
}

std::vector<std::pair<KernelAttr, RealMakeTupleCpuKernelMod::RealMakeTupleFunc>> RealMakeTupleCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeUInt8)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeUInt8),
    &RealMakeTupleCpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt8)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt8),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt16)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt16),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat16)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat16),
    &RealMakeTupleCpuKernelMod::LaunchKernel<float16>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeComplex64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeComplex64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<complex64>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeComplex128)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeComplex128),
    &RealMakeTupleCpuKernelMod::LaunchKernel<complex128>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeBool),
    &RealMakeTupleCpuKernelMod::LaunchKernel<bool>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<float>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kObjectTypeTuple, kNumberTypeUInt8),
    &RealMakeTupleCpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt8),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt16),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat16),
    &RealMakeTupleCpuKernelMod::LaunchKernel<float16>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeComplex64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeComplex64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<complex64>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeComplex128)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeComplex128),
    &RealMakeTupleCpuKernelMod::LaunchKernel<complex128>},
   {KernelAttr()
      .AddAllSameAttr(true)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<double>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    &RealMakeTupleCpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kObjectTypeTuple, kNumberTypeBool),
    &RealMakeTupleCpuKernelMod::LaunchKernel<bool>}};

std::vector<KernelAttr> RealMakeTupleCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RealMakeTupleFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RealMakeTuple, RealMakeTupleCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
