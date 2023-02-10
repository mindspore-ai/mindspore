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

#include "plugin/device/cpu/kernel/non_deterministic_ints_cpu_kernel.h"
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kInputNum = 1;
const uint32_t kInpuDims = 1;
const uint32_t kOutputNum = 1;
const uint32_t kInpuSizes = 2;
}  // namespace

void NonDeterministicIntsCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  input_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  output_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (AnfAlgo::IsShapesDynamic({input_shape})) {
    return;
  }
  if (input_shape[0] < kInpuSizes) {
    MS_EXCEPTION(ValueError) << "The input tensor shape must >= 2.";
  }
  if (input_shape.size() != kInpuDims) {
    MS_EXCEPTION(ValueError) << "The input tensor must be a 1-D tensor.";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "NonDeterministicInts does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
}

bool NonDeterministicIntsCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (output_type_ == kNumberTypeInt32 && input_type_ == kNumberTypeInt32) {
    (void)LaunchKernel<int32_t, int32_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeInt64 && input_type_ == kNumberTypeInt32) {
    (void)LaunchKernel<int64_t, int32_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeInt32 && input_type_ == kNumberTypeInt64) {
    (void)LaunchKernel<int32_t, int64_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeInt64 && input_type_ == kNumberTypeInt64) {
    (void)LaunchKernel<int64_t, int64_t>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "The output data type must be one of int32 or int64.";
  }
  return true;
}

template <typename T1, typename T2>
bool NonDeterministicIntsCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &outputs) {
  auto output = reinterpret_cast<T1 *>(outputs[0]->addr);
  auto input = reinterpret_cast<T2 *>(inputs[0]->addr);
  size_t input_elem_num = inputs[0]->size / sizeof(T2);
  size_t output_elem_num = outputs[0]->size / sizeof(T1);
  ShapeVector out_shape;
  for (size_t i = 0; i < input_elem_num; i++) {
    if (input[i] <= 0) {
      MS_EXCEPTION(ValueError) << "Each dimension must be greater than 0.";
    }
    out_shape.push_back(input[i]);
  }
  auto task = [output](size_t start, size_t end) {
    auto max_data = std::numeric_limits<T1>::max();
    std::default_random_engine seed(time(nullptr));
    std::uniform_int_distribution<T1> u(-max_data, max_data);
    for (size_t i = start; i < end; ++i) {
      output[i] = u(seed);
    }
  };
  CPUKernelUtils::ParallelFor(task, output_elem_num);
  common::AnfAlgo::SetOutputInferTypeAndShape({output_type_}, {out_shape}, cnode_ptr_.lock().get());
  return true;
}

std::vector<std::pair<KernelAttr, NonDeterministicIntsCPUKernelMod::NonDeterministicIntsFunc>>
  NonDeterministicIntsCPUKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsCPUKernelMod::LaunchKernel<int64_t, int64_t>}};

std::vector<KernelAttr> NonDeterministicIntsCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NonDeterministicIntsFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NonDeterministicInts, NonDeterministicIntsCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
