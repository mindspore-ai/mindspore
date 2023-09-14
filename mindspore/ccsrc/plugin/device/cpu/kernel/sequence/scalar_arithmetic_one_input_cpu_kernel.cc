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

#include "plugin/device/cpu/kernel/sequence/scalar_arithmetic_one_input_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <limits>
#include <set>
#include <cmath>
#include <string>
#include <unordered_map>
#include <complex>
#include "mindspore/core/ops/arithmetic_ops.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kScalarUadd = "ScalarUadd";
constexpr auto kScalarUsub = "ScalarUsub";
constexpr auto kScalarLog = "ScalarLog";
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

template <typename T, typename S>
void LogImpl(const T *in_x, S *out) {
  *out = static_cast<S>(log(*in_x));
}

template <typename T, typename S>
void UaddImpl(const T *in_x, S *out) {
  *out = static_cast<S>(*in_x);
}

template <typename T, typename S>
void UsubImpl(const T *in_x, S *out) {
  *out = static_cast<S>(-(*in_x));
}

bool ScalarOneInputCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For kernel '" << kernel_type_ << "' input_num must be 1, but got " << inputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  if (kernel_type_ == kScalarLog) {
    kernel_func_ = log_func_list_[index].second;
  } else {
    kernel_func_ = default_func_list_[index].second;
  }
  return true;
}

int ScalarOneInputCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool ScalarOneInputCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                              const std::vector<AddressPtr> &outputs) {
  using MathImplFunc = std::function<void(const T *x, S *out)>;
  std::unordered_map<std::string, MathImplFunc> func_map = {
    {kScalarUadd, UaddImpl<T, S>}, {kScalarUsub, UsubImpl<T, S>}, {kScalarLog, LogImpl<T, S>}};
  auto iter = func_map.find(kernel_name_);
  if (iter == func_map.end()) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "' don't support. Only support [Uadd, Usub, Log]";
  }
  MathImplFunc compute_func = iter->second;

  T *input_x = GetDeviceAddress<T>(inputs, 0);
  MS_EXCEPTION_IF_NULL(input_x);
  S *output = GetDeviceAddress<S>(outputs, 0);
  MS_EXCEPTION_IF_NULL(output);
  compute_func(input_x, output);
  return true;
}

std::vector<std::pair<KernelAttr, ScalarOneInputCpuKernelMod::ScalarArithmeticFunc>>
  ScalarOneInputCpuKernelMod::log_func_list_ = {{KernelAttr()
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                                                   .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
                                                 &ScalarOneInputCpuKernelMod::LaunchKernel<float, float>},
                                                {KernelAttr()
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
                                                   .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
                                                 &ScalarOneInputCpuKernelMod::LaunchKernel<double, float>},
                                                {KernelAttr()
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                                                   .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
                                                 &ScalarOneInputCpuKernelMod::LaunchKernel<int32_t, float>},
                                                {KernelAttr()
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                   .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
                                                 &ScalarOneInputCpuKernelMod::LaunchKernel<int64_t, float>}};

std::vector<std::pair<KernelAttr, ScalarOneInputCpuKernelMod::ScalarArithmeticFunc>>
  ScalarOneInputCpuKernelMod::default_func_list_ = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat32),
     &ScalarOneInputCpuKernelMod::LaunchKernel<float, float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOutputAttr(kObjectTypeNumber, kNumberTypeFloat64),
     &ScalarOneInputCpuKernelMod::LaunchKernel<double, double>},
    {KernelAttr().AddInputAttr(kObjectTypeNumber, kNumberTypeInt32).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt32),
     &ScalarOneInputCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kObjectTypeNumber, kNumberTypeInt64).AddOutputAttr(kObjectTypeNumber, kNumberTypeInt64),
     &ScalarOneInputCpuKernelMod::LaunchKernel<int64_t, int64_t>}};

std::vector<KernelAttr> ScalarOneInputCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  if (kernel_type_ == kScalarLog) {
    (void)std::transform(log_func_list_.begin(), log_func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, ScalarArithmeticFunc> &item) { return item.first; });
  } else {
    (void)std::transform(default_func_list_.begin(), default_func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, ScalarArithmeticFunc> &item) { return item.first; });
  }
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarUadd,
                                 []() { return std::make_shared<ScalarOneInputCpuKernelMod>(kScalarUadd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarUsub,
                                 []() { return std::make_shared<ScalarOneInputCpuKernelMod>(kScalarUsub); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarLog,
                                 []() { return std::make_shared<ScalarOneInputCpuKernelMod>(kScalarLog); });
}  // namespace kernel
}  // namespace mindspore
