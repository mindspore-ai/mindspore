/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/uniform_cpu_kernel.h"

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
#include <limits>

#include "mindspore/core/ops/uniform.h"
#include "kernel/common_utils.h"
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kUniformInputsNum = 1;
const size_t kUniformOutputsNum = 1;
}  // namespace

uint64_t UniformCpuKernelMod::New64() const {
  const int64_t int64_t_max = (std::numeric_limits<int64_t>::max)();
  std::random_device device("/dev/urandom");
  std::uniform_int_distribution<int64_t> distrib(0, int64_t_max);
  static std::mt19937_64 rng = std::mt19937_64(device());
  return distrib(rng);
}

void UniformCpuKernelMod::InitPhiloxRandom(int64_t seed, int64_t offset) {
  if (seed == 0 && offset == 0) {
    seed = SizeToLong(New64());
    offset = SizeToLong(New64());
  }
  generator_ = random::PhiloxRandom(seed, offset);
}

float UniformCpuKernelMod::RandFloat() {
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

uint32_t UniformCpuKernelMod::GenerateSingle() {
  if (used_result_index_ == random::PhiloxRandom::kResultElementCount) {
    unused_results_ = generator_();
    used_result_index_ = 0;
  }
  return unused_results_[used_result_index_++];
}

bool UniformCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto op = std::dynamic_pointer_cast<ops::Uniform>(base_operator);
  kernel_name_ = op->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  kernel_ptr_ = std::make_shared<ops::Uniform>(base_operator->GetPrim());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Uniform does not support this kernel data type: " << kernel_attr;
  }
  from_ = op->get_from();
  to_ = op->get_to();
  int64_t seed = op->get_seed();
  int64_t offset = op->get_offset();
  InitPhiloxRandom(seed, offset);
  if (from_ > to_) {
    MS_LOG(ERROR) << "For Uniform, 'minval' must <= 'maxval', but got 'minval'=" << from_ << " ,'maxval'=" << to_;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UniformCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  input_elements_ = SizeToLong(SizeOf(inputs.at(kIndex0)->GetShapeVector()));
  return ret;
}

template <typename T>
bool UniformCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniformInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniformOutputsNum, kernel_name_);

  auto y = reinterpret_cast<T *>(outputs[0]->addr);
  for (int64_t i = 0; i < input_elements_; i++) {
    y[i] = static_cast<T>(RandFloat() * (to_ - from_) + from_);
  }

  return true;
}

std::vector<std::pair<KernelAttr, UniformCpuKernelMod::UniformFunc>> UniformCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &UniformCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &UniformCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &UniformCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> UniformCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UniformFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Uniform, UniformCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
