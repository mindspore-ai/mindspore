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
constexpr uint32_t kNumInput = 1;
constexpr uint32_t kNumOutput = 1;
constexpr auto kAttrMean = "mean";
constexpr auto kAttrStd = "std";
}  // namespace

bool LogNormalReverseCpuKernel::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_ERROR_IF_NULL(prim);
  input_mean_ = GetValue<float>(prim->GetAttr(kAttrMean));
  input_std_ = GetValue<float>(prim->GetAttr(kAttrStd));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int LogNormalReverseCpuKernel::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_dtype_ = inputs[kIndex0]->GetDtype();
  return KRET_OK;
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
