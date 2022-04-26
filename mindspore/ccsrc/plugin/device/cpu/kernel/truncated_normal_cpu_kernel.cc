/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/truncated_normal_cpu_kernel.h"
#include <cmath>
#include <ctime>
#include <random>
#include <utility>
#include <vector>
#include <algorithm>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
const int32_t kMax = 2;
const uint32_t kInputNum = 1;
const uint32_t kInputDims = 1;
const uint32_t kOutputNum = 1;
const uint32_t kInputSizes = 2;
}  // namespace

void TruncatedNormalCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  output_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  seed_ = static_cast<size_t>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed"));
  seed2_ = static_cast<size_t>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed2"));
  if (input_shape[0] < kInputSizes) {
    MS_EXCEPTION(ValueError) << "The input tensor shape must >= 2.";
  }
  if (input_shape.size() != kInputDims) {
    MS_EXCEPTION(ValueError) << "The input tensor must be a 1-D tensor.";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "TruncatedNormal does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
}

bool TruncatedNormalCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (input_type_ == kNumberTypeInt32 && output_type_ == kNumberTypeFloat16) {
    LaunchKernel<int32_t, float16, float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt32 && output_type_ == kNumberTypeFloat32) {
    LaunchKernel<int32_t, float, float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt32 && output_type_ == kNumberTypeFloat64) {
    LaunchKernel<int32_t, double, double>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt64 && output_type_ == kNumberTypeFloat16) {
    LaunchKernel<int64_t, float16, float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt64 && output_type_ == kNumberTypeFloat32) {
    LaunchKernel<int64_t, float, float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt64 && output_type_ == kNumberTypeFloat64) {
    LaunchKernel<int64_t, double, double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "The output data type must be one of float16, float32 and float64.";
  }
  return true;
}

template <typename T1, typename T2, typename T3>
bool TruncatedNormalCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  auto input = reinterpret_cast<T1 *>(inputs[0]->addr);
  size_t input_elem_num = inputs[0]->size / sizeof(T1);
  for (size_t i = 0; i < input_elem_num; i++) {
    if (input[i] <= 0) {
      MS_EXCEPTION(ValueError) << "Each dimension must be greater than zero.";
    }
  }

  auto output = reinterpret_cast<T2 *>(outputs[0]->addr);
  size_t output_elem_num = outputs[0]->size / sizeof(T2);
  std::random_device rd;
  seedc_ = seed2_ != 0 ? seed2_ : (seed_ != 0 ? seed_ : rd());
  std::default_random_engine final_seed(seedc_);
  if (seed_ != 0 || seed2_ != 0) {
    flag_ = false;
  }

  std::normal_distribution<T3> dis(0, 1);
  auto task = [&](size_t start, size_t end) {
    for (size_t j = start; j < end;) {
      auto data = dis(final_seed);
      if (data >= -kMax && data <= kMax) {
        output[j++] = static_cast<T2>(data);
      }
    }
  };
  if (flag_) {
    CPUKernelUtils::ParallelFor(task, output_elem_num);
  } else {
    for (size_t i = 0; i < output_elem_num;) {
      auto data = dis(final_seed);
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
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
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
}  // namespace kernel
}  // namespace mindspore
