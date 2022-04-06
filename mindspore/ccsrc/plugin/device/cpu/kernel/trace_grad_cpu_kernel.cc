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
#include "plugin/device/cpu/kernel/trace_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 2;
constexpr size_t kInputShapeDim = 2;
constexpr size_t kOutputNum = 1;
}  // namespace

void TraceGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (input_shape_.size() != kInputShapeDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input tensor's dimension should be " << kInputShapeDim
                      << ", but got " << input_shape_.size();
  }
  values_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
}

bool TraceGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  switch (values_type_) {
    case kNumberTypeInt8:
      LaunchKernel<int8_t>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      LaunchKernel<int16_t>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      LaunchKernel<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      LaunchKernel<int64_t>(inputs, outputs);
      break;
    case kNumberTypeUInt8:
      LaunchKernel<uint8_t>(inputs, outputs);
      break;
    case kNumberTypeUInt16:
      LaunchKernel<uint16_t>(inputs, outputs);
      break;
    case kNumberTypeUInt32:
      LaunchKernel<uint32_t>(inputs, outputs);
      break;
    case kNumberTypeUInt64:
      LaunchKernel<uint64_t>(inputs, outputs);
      break;
    case kNumberTypeFloat16:
      LaunchKernel<float16>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      LaunchKernel<float>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      LaunchKernel<double>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "Unsupported input data type.";
  }
  return true;
}

template <typename T>
void TraceGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto grad = reinterpret_cast<T *>(inputs[0]->addr);
  auto shape = reinterpret_cast<int64_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t min_size = std::min(shape[0], shape[1]);
  for (size_t i = 0; i < min_size; ++i) {
    *(output_addr + i * shape[1] + i) = *grad;
  }
}
}  // namespace kernel
}  // namespace mindspore
