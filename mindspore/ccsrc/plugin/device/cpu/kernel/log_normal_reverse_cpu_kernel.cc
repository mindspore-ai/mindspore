/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "mindspore/ccsrc/plugin/device/cpu/kernel/log_normal_reverse_cpu_kernel.h"
#include <cmath>
#include <random>
#include <ctime>
#include <iostream>
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Core"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kNumInput = 1;
const uint32_t kNumOutput = 1;
}  // namespace

void LogNormalReverseCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_dtype_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (input_dtype_ != kNumberTypeFloat32 && input_dtype_ != kNumberTypeFloat16) {
    if (input_dtype_ != kNumberTypeFloat64) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << ", the datatype of the input1 not support, support datatype: float16, float32, float64.";
    }
  }
  if (input_dtype_ != output_dtype_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the data type of the input does not match the data type of the output.";
  }
  input_mean_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "mean");
  input_std_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "std");
}

bool LogNormalReverseCpuKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNumInput, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNumOutput, kernel_name_);
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernelFloat<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    LaunchKernelFloat<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "LogNormalReverse kernel data type "
                      << "float16, float32"
                      << " were support.";
  }
  return true;
}

template <typename T>
void LogNormalReverseCpuKernel::LaunchKernelFloat(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);

  size_t elem_num = inputs[0]->size / sizeof(T);

  static std::default_random_engine random_engine(time(0));
  static std::normal_distribution<float> normal_value(input_mean_, input_std_);

  for (size_t i = 0; i < elem_num; i++) {
    output[i] = static_cast<T>(std::exp(normal_value(random_engine)));
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogNormalReverse, LogNormalReverseCpuKernel);
}  // namespace kernel
}  // namespace mindspore
