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

void CompareAndBitpackCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
  input0_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  input1_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex1);
  if (input1_shape_.size() != 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input[threshold] must be 0D"
                             << ", but got shape " << Vector2Str(input1_shape_);
  }

  if (input0_shape_.size() == 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input[x] must not be 0D"
                             << ", but got shape " << Vector2Str(input0_shape_);
  }

  int64_t last_dim_index = input0_shape_[input0_shape_.size() - 1];
  int32_t divisible_num = 8;
  if (last_dim_index % divisible_num != 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the inner dimension of input[x] should be divisible by 8"
                             << ", but got shape: " << Vector2Str(input0_shape_);
  }
}

bool CompareAndBitpackCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCompareAndBitpackInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCompareAndBitpackOutputsNum, kernel_name_);
  uint32_t res = true;
  switch (dtype_) {
    case kNumberTypeFloat16:
      res = LaunchKernel<float16>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      res = LaunchKernel<float>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      res = LaunchKernel<double>(inputs, outputs);
      break;
    case kNumberTypeBool:
      res = LaunchKernel<bool>(inputs, outputs);
      break;
    case kNumberTypeInt8:
      res = LaunchKernel<int8_t>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      res = LaunchKernel<int16_t>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      res = LaunchKernel<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      res = LaunchKernel<int64_t>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "CompareAndBitpack invalid input type " << TypeIdLabel(dtype_) << " which is not supported.";
  }
  return res;
}

template <typename T>
bool CompareAndBitpackCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
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
      const int64_t block = *reinterpret_cast<const int64_t *>(input0_data + 8 * i);
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

std::vector<KernelAttr> CompareAndBitpackCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CompareAndBitpack, CompareAndBitpackCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
