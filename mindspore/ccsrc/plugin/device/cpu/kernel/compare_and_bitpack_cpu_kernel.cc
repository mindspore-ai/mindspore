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

#include "plugin/device/cpu/kernel/compare_and_bitpack_cpu_kernel.h"
#include <algorithm>
#include "unsupported/Eigen/CXX11/Tensor"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCompareAndBitpackInputsNum = 2;
constexpr size_t kCompareAndBitpackOutputsNum = 1;
}  // namespace

bool CompareAndBitpackCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCompareAndBitpackInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCompareAndBitpackOutputsNum, kernel_name_);
  dtype_ = inputs[kIndex0]->GetDtype();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

template <typename T>
bool CompareAndBitpackCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  T *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input1 = reinterpret_cast<T *>(inputs[1]->addr);
  uint8_t *output = reinterpret_cast<uint8_t *>(outputs[0]->addr);
  int64_t data_num = SizeToLong(outputs[0]->size);
  T thresh = *input1;
  const int64_t shift_num1 = 1;
  const int64_t shift_num2 = 2;
  const int64_t shift_num3 = 3;
  const int64_t shift_num4 = 4;
  const int64_t shift_num5 = 5;
  const int64_t shift_num6 = 6;
  const int64_t shift_num7 = 7;
  const int64_t shift_num8 = 8;
  if (dtype_ == kNumberTypeBool) {
    // Specialization for bool on systems where sizeof(bool) == 1.
    for (int64_t i = 0; i < data_num; ++i) {
      uint8_t *out = output + i;
      bool *input0_data = reinterpret_cast<bool *>(inputs[0]->addr);
      int64_t block = *reinterpret_cast<int64_t *>(input0_data + 8 * i);
      *out = ((((block & (1LL << (shift_num7 * shift_num8))) >> (shift_num7 * shift_num8 - shift_num7))) |
              (((block & (1LL << (shift_num6 * shift_num8))) >> (shift_num6 * shift_num8 - shift_num6))) |
              (((block & (1LL << (shift_num5 * shift_num8))) >> (shift_num5 * shift_num8 - shift_num5))) |
              (((block & (1LL << (shift_num4 * shift_num8))) >> (shift_num4 * shift_num8 - shift_num4))) |
              (((block & (1LL << (shift_num3 * shift_num8))) >> (shift_num3 * shift_num8 - shift_num3))) |
              (((block & (1LL << (shift_num2 * shift_num8))) >> (shift_num2 * shift_num8 - shift_num2))) |
              (((block & (1LL << shift_num8)) >> (shift_num1 * shift_num8 - shift_num1))) | (((block & (1LL)))));
    }
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      uint8_t *out = output + i;
      const T *input = input0 + 8 * i;
      *out = ((((input[kIndex0] > thresh) << shift_num7)) | (((input[kIndex1] > thresh) << shift_num6)) |
              (((input[kIndex2] > thresh) << shift_num5)) | (((input[kIndex3] > thresh) << shift_num4)) |
              (((input[kIndex4] > thresh) << shift_num3)) | (((input[kIndex5] > thresh) << shift_num2)) |
              (((input[kIndex6] > thresh) << shift_num1)) | (((input[kIndex7] > thresh))));
    }
  }

  return true;
}

const std::vector<std::pair<KernelAttr, CompareAndBitpackCpuKernelMod::KernelRunFunc>>
  &CompareAndBitpackCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, CompareAndBitpackCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackCpuKernelMod::LaunchKernel<int64_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CompareAndBitpack, CompareAndBitpackCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
