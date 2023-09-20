/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/multinomial_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <functional>
#include <utility>
#include "kernel/ops_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/philox_random.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kWorkspaceNum = 1;
constexpr uint32_t kOutputNum = 1;

// clang-format off
#define ADD_KERNEL(prob_dtype, prob_type)                                                                              \
  {KernelAttr().AddInputAttr(kNumberType##prob_dtype).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),  \
  &MultinomialCpuKernelMod::LaunchKernel<prob_type, int32_t>},                                                         \
  {KernelAttr().AddInputAttr(kNumberType##prob_dtype).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),  \
  &MultinomialCpuKernelMod::LaunchKernel<prob_type, int64_t>},                                                         \
  {KernelAttr().AddInputAttr(kNumberType##prob_dtype).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),  \
  &MultinomialCpuKernelMod::LaunchKernel<prob_type, int32_t>},                                                         \
  {KernelAttr().AddInputAttr(kNumberType##prob_dtype).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),  \
  &MultinomialCpuKernelMod::LaunchKernel<prob_type, int64_t>}
// clang-format on
}  // namespace

bool MultinomialCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  output_dtype_ = outputs[0]->GetDtype();
  input0_dtype_ = inputs[0]->GetDtype();
  input1_dtype_ = inputs[1]->GetDtype();
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("seed2")));
  uint64_t init_seed = random::GetSeed(seed, seed2);
  rng_.seed(init_seed);
  return true;
}

int MultinomialCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  int ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  int64_t elem_num = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
  (void)workspace_size_list_.emplace_back(elem_num * sizeof(TypeIdToType(input0_dtype_)));
  return ret;
}

void MultinomialCpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T_in, typename T_out>
bool MultinomialCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  CHECK_KERNEL_WORKSPACE_SIZE(workspace.size(), kWorkspaceNum, kernel_name_);

  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  MS_EXCEPTION_IF_NULL(workspace[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);

  auto *input_tensor = reinterpret_cast<T_in *>(inputs[0]->addr);
  int num_sample = reinterpret_cast<int *>(inputs[1]->addr)[0];
  auto *output = reinterpret_cast<T_out *>(outputs[0]->addr);
  auto *cumulative_value = reinterpret_cast<T_in *>(workspace[0]->addr);

  // check num_samples nonnegative
  if (num_sample < 0.0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "' num_samples should be a nonnegative number, but got "
                             << num_sample << ".";
  }

  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(cumulative_value);

  size_t num_row = 1;
  size_t num_shape = 2;
  if (input_shape_.size() == num_shape) {
    num_row = static_cast<size_t>(input_shape_[0]);
  }
  size_t num_col = static_cast<size_t>(input_shape_[input_shape_.size() - 1]);

  for (size_t i = 0; i < num_row; ++i) {
    // Compute the cumulative array.
    cumulative_value[i * num_col] = input_tensor[i * num_col];
    for (size_t j = 1; j < IntToSize(num_col); ++j) {
      size_t index = i * num_col + j;
      cumulative_value[index] = cumulative_value[index - 1] + input_tensor[index];
    }

    // Normalize the cumulative array.
    auto sum = cumulative_value[(i + 1) * num_col - 1];
    if (sum != static_cast<T_in>(0.0)) {
      for (size_t k = 0; k < IntToSize(num_col); ++k) {
        size_t index = i * num_col + k;
        cumulative_value[index] /= sum;
      }
    }

    // Initialize random generator.
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Sample data from cumulative array.
    for (int64_t n = 0; n < IntToLong(num_sample); ++n) {
      auto rand_prob = static_cast<T_in>(dist(rng_));
      int64_t begin = 0;
      int64_t end = SizeToLong(num_col - 1);

      while (end - begin > 0) {
        int64_t pivot = begin + (end - begin) / 2;
        auto pivot_prob = cumulative_value[i * num_col + LongToSize(pivot)];
        if (pivot_prob > rand_prob) {
          end = pivot;
        } else {
          begin = pivot + 1;
        }
      }
      output[i * static_cast<size_t>(num_sample) + static_cast<size_t>(n)] = begin;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, MultinomialCpuKernelMod::MultinomialFunc>> MultinomialCpuKernelMod::func_list_ = {
  ADD_KERNEL(Float16, float16), ADD_KERNEL(Float32, float),   ADD_KERNEL(Float64, double), ADD_KERNEL(Int8, int8_t),
  ADD_KERNEL(Int16, int16_t),   ADD_KERNEL(Int32, int32_t),   ADD_KERNEL(Int64, int64_t),  ADD_KERNEL(UInt8, uint8_t),
  ADD_KERNEL(UInt16, uint16_t), ADD_KERNEL(UInt32, uint32_t), ADD_KERNEL(UInt64, uint64_t)};

std::vector<KernelAttr> MultinomialCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MultinomialFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Multinomial, MultinomialCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
