/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * UnBernoulli required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/cpu/kernel/bernoulli_cpu_kernel.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <functional>
#include <random>

#include "mindspore/core/ops/bernoulli.h"
#include "kernel/common_utils.h"
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kBernoulliInputsNum = 2;
const size_t kBernoulliOutputsNum = 1;
const int64_t kBernoulliDefaultSeed = -1;
const int64_t kBernoulliDefaultOffset = 0;
}  // namespace

uint64_t BernoulliCpuKernelMod::New64() const {
  std::random_device device("/dev/urandom");
  static std::mt19937_64 rng = std::mt19937_64(device());
  return (rng)();
}

void BernoulliCpuKernelMod::InitMSPhiloxRandom(int64_t seed_, int64_t offset_) {
  if (seed_ == kBernoulliDefaultSeed && offset_ == kBernoulliDefaultOffset) {
    seed_ = SizeToUlong(New64());
    offset_ = SizeToUlong(New64());
  }
  generator_ = random::PhiloxRandom(seed_, offset_);
}

float BernoulliCpuKernelMod::RandFloat() {
  uint32_t x = GenerateSingle();
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;

  float result;
  int ret = memcpy_s(&result, sizeof(result), &val, sizeof(val));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "The memcpy error, errorno(" << ret << ")";
  }
  return result - 1.0f;
}

uint32_t BernoulliCpuKernelMod::GenerateSingle() {
  if (used_result_index_ == random::PhiloxRandom::kResultElementCount) {
    unused_results_ = generator_();
    used_result_index_ = 0;
  }
  return unused_results_[used_result_index_++];
}

bool BernoulliCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto op = std::dynamic_pointer_cast<ops::Bernoulli>(base_operator);
  kernel_name_ = op->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  kernel_ptr_ = std::make_shared<ops::Bernoulli>(base_operator->GetPrim());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Bernoulli does not support this kernel data type: " << kernel_attr;
  }
  seed_ = op->get_seed();
  offset_ = kBernoulliDefaultOffset;
  kernel_func_ = func_list_[index].second;
  return true;
}

int BernoulliCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  p_shape_ = inputs.at(kIndex1)->GetShapeVector();
  return ret;
}

template <typename T, typename S>
bool BernoulliCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBernoulliInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBernoulliOutputsNum, kernel_name_);

  InitMSPhiloxRandom(seed_, offset_);

  input_elements_nums = std::accumulate(x_shape_.begin(), x_shape_.end(), int64_t(1), std::multiplies<int64_t>());
  auto p = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  auto y = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  int64_t p_dims = static_cast<int64_t>(p_shape_.size());
  int64_t x_dims = static_cast<int64_t>(x_shape_.size());
  auto p_num = p_dims == 0 ? 1 : p_shape_[0];
  if (p_num == 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', p can't be empty.";
  } else if (p_dims == 0 || (p_dims == 1 && p_num == 1)) {
    if (*p > 1 || *p < 0) {
      MS_EXCEPTION(ValueError) << "For bernoulli, p should be in [0, 1], but got " << *p << ".";
    }
    for (int i = 0; i < input_elements_nums; i++) {
      float a = RandFloat();
      if (a > *p) {
        y[i] = 0;
      } else {
        y[i] = 1;
      }
    }
  } else {
    if (p_dims != x_dims) {
      MS_EXCEPTION(ValueError) << "For bernoulli, the shape of p is different from the shape of x.";
    } else {
      for (int i = 0; i < p_dims; i++) {
        if (p_shape_[i] != x_shape_[i]) {
          MS_EXCEPTION(ValueError) << "For bernoulli, the shape of p is different from the shape of x.";
        }
      }
    }
    for (int i = 0; i < input_elements_nums; i++) {
      if (p[i] > 1 || p[i] < 0) {
        MS_EXCEPTION(ValueError) << "For bernoulli, p should be in [0, 1].";
      }
    }
    for (int i = 0; i < input_elements_nums; i++) {
      if (RandFloat() > p[i]) {
        y[i] = 0;
      } else {
        y[i] = 1;
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, BernoulliCpuKernelMod::BernoulliFunc>> BernoulliCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &BernoulliCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
   &BernoulliCpuKernelMod::LaunchKernel<double, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt8),
   &BernoulliCpuKernelMod::LaunchKernel<int8_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt16),
   &BernoulliCpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &BernoulliCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &BernoulliCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
   &BernoulliCpuKernelMod::LaunchKernel<uint8_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   &BernoulliCpuKernelMod::LaunchKernel<bool, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
   &BernoulliCpuKernelMod::LaunchKernel<float, double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &BernoulliCpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt8),
   &BernoulliCpuKernelMod::LaunchKernel<int8_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt16),
   &BernoulliCpuKernelMod::LaunchKernel<int16_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &BernoulliCpuKernelMod::LaunchKernel<int32_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &BernoulliCpuKernelMod::LaunchKernel<int64_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
   &BernoulliCpuKernelMod::LaunchKernel<uint8_t, double>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
   &BernoulliCpuKernelMod::LaunchKernel<bool, double>}};

std::vector<KernelAttr> BernoulliCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BernoulliFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Bernoulli, BernoulliCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
