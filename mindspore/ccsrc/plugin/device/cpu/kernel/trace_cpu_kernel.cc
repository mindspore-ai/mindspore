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
#include "plugin/device/cpu/kernel/trace_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kInputDim = 2;
constexpr size_t kOutputNum = 1;
}  // namespace

void TraceCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto device_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (IsDynamic(device_shape)) {
    return;
  }
  input_shape_ = Convert2SizeT(device_shape);
  values_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (input_shape_.size() != kInputDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input tensor's dimension should be " << kInputDim
                      << ", but got " << input_shape_.size();
  }
}

bool TraceCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  switch (values_type) {
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
void TraceCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  size_t min_size = std::min(input_shape_[0], input_shape_[1]);
  (void)memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  for (size_t i = 0; i < min_size; ++i) {
    *output_addr += *(input_addr + i * input_shape_[1] + i);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Trace, TraceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
