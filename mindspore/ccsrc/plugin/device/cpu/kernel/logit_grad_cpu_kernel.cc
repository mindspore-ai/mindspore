/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/logit_grad_cpu_kernel.h"
#include <functional>
#include <limits>
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLogitGradInputsNum = 2;
constexpr size_t kLogitGradOutputsNum = 1;
}  // namespace

void LogitGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  // grad_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kLogitGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kLogitGradOutputsNum, kernel_name_);
  MS_EXCEPTION_IF_NULL(kernel_node);
}

bool LogitGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
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

bool LogitGradCpuKernelMod::LaunchKernelHalf(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto eps = common::AnfAlgo::GetNodeAttr<float>(node_, "eps");
  int64_t input_elem_num = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<int>());
  float16 *grad = reinterpret_cast<float16 *>(inputs[0]->addr);
  float16 *input = reinterpret_cast<float16 *>(inputs[1]->addr);
  float16 *output = reinterpret_cast<float16 *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size;
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (int64_t i = 0; i < input_elem_num; i++) {
      output[i] = (input[i] < float16(0) || input[i] > float16(1))
                    ? float16(std::numeric_limits<float>::quiet_NaN())
                    : float16(static_cast<float>(grad[i]) / static_cast<float>(input[i]) /
                              (static_cast<float>(1) - static_cast<float>(input[i])));
    }
  } else {
    for (int64_t i = 0; i < input_elem_num; i++) {
      output[i] = (static_cast<float>(input[i]) < static_cast<float>(eps) ||
                   static_cast<float>(input[i]) > static_cast<float>(1 - eps))
                    ? float16(0)
                    : float16(static_cast<float>(grad[i]) / static_cast<float>(input[i]) /
                              (static_cast<float>(1) - static_cast<float>(input[i])));
    }
  }
  return true;
}

template <typename T>
bool LogitGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto eps = common::AnfAlgo::GetNodeAttr<float>(node_, "eps");
  int64_t input_elem_num = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<int>());
  T *grad = reinterpret_cast<T *>(inputs[0]->addr);
  T *input = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size;
  if (memset_s(output, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (eps < 0) {
    for (int64_t i = 0; i < input_elem_num; i++) {
      output[i] = (input[i] < T(0) || input[i] > T(1)) ? std::numeric_limits<T>::quiet_NaN()
                                                       : (grad[i] / input[i] / (T(1) - input[i]));
    }
  } else {
    for (int64_t i = 0; i < input_elem_num; i++) {
      output[i] = (input[i] < static_cast<T>(eps) || input[i] > T(1) - static_cast<T>(eps))
                    ? T(0)
                    : (grad[i] / input[i] / (T(1) - input[i]));
    }
  }
  return true;
}

std::vector<KernelAttr> LogitGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LogitGrad, LogitGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
