/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/logit_cpu_kernel.h"
#include <functional>
#include <limits>
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogitInputsNum = 1;
constexpr size_t kLogitOutputsNum = 1;
}  // namespace

void LogitCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kLogitInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kLogitOutputsNum, kernel_name_);
  MS_EXCEPTION_IF_NULL(kernel_node);
}

bool LogitCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernelHalf(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the dtype of input should be float16, float32 or float64, but got "
                            << TypeIdToType(input_dtype_)->ToString();
  }
  return true;
}

bool LogitCpuKernelMod::LaunchKernelHalf(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto eps = common::AnfAlgo::GetNodeAttr<float>(node_, "eps");
  int64_t input_elem_num = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<int>());
  float16 *input = reinterpret_cast<float16 *>(inputs[0]->addr);
  float16 *output = reinterpret_cast<float16 *>(outputs[0]->addr);
  float16 one = float16(1);
  float16 up_bound = float16(static_cast<float>(1) - static_cast<float>(eps));
  size_t output_size = outputs[0]->size;
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (int64_t i = 0; i < input_elem_num; i++) {
      float16 x = input[i];
      output[i] = log(x / (one - x));
    }
  } else {
    for (int64_t i = 0; i < input_elem_num; i++) {
      float16 z;
      float16 x = input[i];
      z = x < static_cast<float16>(eps) ? static_cast<float16>(eps) : (x > up_bound ? up_bound : x);
      output[i] = log(z / (one - z));
    }
  }
  return true;
}

template <typename T>
bool LogitCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto eps = common::AnfAlgo::GetNodeAttr<float>(node_, "eps");
  int64_t input_elem_num = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<int>());
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  T one = T(1);
  T up_bound = static_cast<T>(1) - static_cast<T>(eps);
  size_t output_size = outputs[0]->size;
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (int64_t i = 0; i < input_elem_num; i++) {
      T x = input[i];
      output[i] = log(x / (one - x));
    }
  } else {
    for (int64_t i = 0; i < input_elem_num; i++) {
      T z;
      T x = input[i];
      z = x < static_cast<T>(eps) ? static_cast<T>(eps) : (x > up_bound ? up_bound : x);
      output[i] = log(z / (one - z));
    }
  }
  return true;
}

std::vector<KernelAttr> LogitCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Logit, LogitCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
