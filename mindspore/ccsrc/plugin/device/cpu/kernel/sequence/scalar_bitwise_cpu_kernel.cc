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

#include "plugin/device/cpu/kernel/sequence/scalar_bitwise_cpu_kernel.h"
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
constexpr auto kScalarBitwiseAnd = "bit_and";
constexpr auto kScalarBitwiseOr = "bit_or";
constexpr size_t kInputNum = 2;
constexpr size_t kInputx = 0;
constexpr size_t kInputy = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

template <typename T, typename S, typename N>
void AddImpl(const T *in_x, const S *in_y, N *out) {
  N x = static_cast<N>(*in_x);
  N y = static_cast<N>(*in_y);
#ifndef _MSC_VER
  if constexpr (std::is_integral<N>::value && std::is_signed<N>::value) {
    if (__builtin_add_overflow(x, y, out)) {
      MS_EXCEPTION(ValueError) << "For prim ScalarAdd Overflow of the sum of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return;
  }
#endif
  *out = x + y;
}

bool ScalarBitwiseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For kernel '" << kernel_type_ << "' input_num must be 2, but got " << inputs.size();
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

int ScalarBitwiseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T, typename S, typename N>
bool ScalarBitwiseCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);

  T *input_x = GetDeviceAddress<T>(inputs, kInputx);
  S *input_y = GetDeviceAddress<S>(inputs, kInputy);
  N *output = GetDeviceAddress<N>(outputs, 0);
  auto x = static_cast<N>(*input_x);
  auto y = static_cast<N>(*input_y);
  if (kernel_type_ == kScalarBitwiseAnd) {
    *output = x & y;
  } else {
    *output = x | y;
  }
  return true;
}

#define ADD_KERNEL(x_dtype, y_dtype, out_dtype, x_type, y_type, out_type) \
  {                                                                       \
    KernelAttr()                                                          \
      .AddInputAttr(kObjectTypeNumber, kNumberType##x_dtype)              \
      .AddInputAttr(kObjectTypeNumber, kNumberType##y_dtype)              \
      .AddOutputAttr(kObjectTypeNumber, kNumberType##out_dtype),          \
      &ScalarBitwiseCpuKernelMod::LaunchKernel<x_type, y_type, out_type>  \
  }

std::vector<std::pair<KernelAttr, ScalarBitwiseCpuKernelMod::ScalarBitwiseFunc>> ScalarBitwiseCpuKernelMod::func_list_ =
  {ADD_KERNEL(Int32, Int32, Int32, int32_t, int32_t, int32_t),
   ADD_KERNEL(Int32, Int64, Int64, int32_t, int64_t, int64_t),
   ADD_KERNEL(Int32, Bool, Int32, int32_t, bool, int32_t),
   ADD_KERNEL(Int64, Int64, Int64, int64_t, int64_t, int64_t),
   ADD_KERNEL(Int64, Int32, Int64, int64_t, int32_t, int64_t),
   ADD_KERNEL(Int64, Bool, Int64, int64_t, bool, int64_t),
   ADD_KERNEL(Bool, Int64, Int64, bool, int64_t, int64_t),
   ADD_KERNEL(Bool, Int32, Int32, bool, int32_t, int32_t),
   ADD_KERNEL(Bool, Bool, Bool, bool, bool, bool)};

std::vector<KernelAttr> ScalarBitwiseCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ScalarBitwiseFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, bit_and,
                                 []() { return std::make_shared<ScalarBitwiseCpuKernelMod>(kScalarBitwiseAnd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, bit_or,
                                 []() { return std::make_shared<ScalarBitwiseCpuKernelMod>(kScalarBitwiseOr); });
}  // namespace kernel
}  // namespace mindspore
