/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/range_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kRangeInputsNum = 3;
constexpr size_t kRangeOutputsNum = 1;

template <typename T>
T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}
}  // namespace

void RangeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  is_need_retrieve_output_shape_ = true;
}

bool RangeCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRangeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRangeOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input must be int or float, but got "
                      << TypeIdLabel(dtype_);
  }
  if (!node_wpt_.expired()) {
    auto node = node_wpt_.lock();
    if (!node) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node;
    }
    ShapeVector out_shape{SizeToLong(output_size_)};
    TypeId out_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
    common::AnfAlgo::SetOutputInferTypeAndShape({out_type}, {out_shape}, node.get());
  }
  return true;
}

template <typename T>
void RangeCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto start = reinterpret_cast<T *>(inputs[0]->addr)[0];
  auto limit = reinterpret_cast<T *>(inputs[1]->addr)[0];
  auto delta = reinterpret_cast<T *>(inputs[2]->addr)[0];
  if (delta == static_cast<T>(0)) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the delta can not be 0.";
  }

  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t max_size = outputs[0]->size / sizeof(T);
  if (Sign(static_cast<float>(delta)) * Sign(static_cast<float>(limit - start)) >= 0) {
    if (std::is_integral<T>::value) {
      output_size_ = static_cast<size_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
    } else {
      output_size_ = static_cast<size_t>(std::ceil((limit - start) / delta));
    }
    if (output_size_ > max_size) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the output element number exceeds the maximum number.";
    }
    for (size_t index = 0; index < output_size_; index++) {
      output[index] = delta * index + start;
    }
  } else {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", upper bound and larger bound inconsistent with step sign.";
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Range, RangeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
