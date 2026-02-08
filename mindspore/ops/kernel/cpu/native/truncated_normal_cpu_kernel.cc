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

#include "kernel/cpu/native/truncated_normal_cpu_kernel.h"
#include <cmath>
#include <ctime>
#include <utility>
#include <vector>
#include <algorithm>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mindspore/ops/infer/truncated_normal.h"

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
namespace truncated_normal_cpu {
namespace {
const int32_t kMax = 2;
const uint32_t kInputNum = 1;
const uint32_t kInputDims = 1;
const uint32_t kOutputNum = 1;
const uint32_t kInputSizes = 2;
}  // namespace

bool TruncatedNormalCPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed2")));
  if (seed != 0 || seed2 != 0) {
    flag_ = false;
  }
  uint64_t init_seed = random::GetSeed(seed, seed2);
  rng_.seed(init_seed);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "TruncatedNormal does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

int TruncatedNormalCPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto shape_input = inputs[kIndex0]->GetShapeVector();
  if (shape_input.size() != kInputDims) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', The input tensor must be a 1-D tensor.";
  }
  if (shape_input[kIndex0] < kInputSizes) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input tensor shape must >= 2, but got "
                             << shape_input[kIndex0];
  }
  input_type_ = inputs[kIndex0]->dtype_id();
  output_type_ = outputs[kIndex0]->dtype_id();
  return KRET_OK;
}

bool TruncatedNormalCPUKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                         const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  MS_ERROR_IF_NULL(kernel_func_);
  return kernel_func_(this, inputs, outputs);
}

template <typename T1, typename T2, typename T3>
bool TruncatedNormalCPUKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<T1>(inputs, kIndex0);
  size_t input_elem_num = inputs[0]->size() / sizeof(T1);
  for (size_t i = 0; i < input_elem_num; i++) {
    if (input[i] <= 0) {
      MS_EXCEPTION(ValueError) << "Each dimension must be greater than zero.";
    }
  }

  auto output = GetDeviceAddress<T2>(outputs, kIndex0);
  size_t output_elem_num = outputs[0]->size() / sizeof(T2);

  std::normal_distribution<double> dis(0, 1);
  auto task = [&](size_t start, size_t end) {
    for (size_t j = start; j < end;) {
      auto data = dis(rng_);
      if (data >= -kMax && data <= kMax) {
        output[j++] = static_cast<T2>(data);
      }
    }
  };
  if (flag_) {
    CPUKernelUtils::ParallelFor(task, output_elem_num);
  } else {
    for (size_t i = 0; i < output_elem_num;) {
      auto data = dis(rng_);
      if (data >= -kMax && data <= kMax) {
        output[i++] = static_cast<T2>(data);
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, TruncatedNormalCPUKernelMod::TruncatedNormalFunc>>
  TruncatedNormalCPUKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &TruncatedNormalCPUKernelMod::LaunchKernel<int32_t, float16, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &TruncatedNormalCPUKernelMod::LaunchKernel<int32_t, float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &TruncatedNormalCPUKernelMod::LaunchKernel<int32_t, double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &TruncatedNormalCPUKernelMod::LaunchKernel<int64_t, float16, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &TruncatedNormalCPUKernelMod::LaunchKernel<int64_t, float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &TruncatedNormalCPUKernelMod::LaunchKernel<int64_t, double, double>}};

std::vector<KernelAttr> TruncatedNormalCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, TruncatedNormalFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TruncatedNormal, TruncatedNormalCPUKernelMod);
}  // namespace truncated_normal_cpu
}  // namespace kernel
}  // namespace mindspore
