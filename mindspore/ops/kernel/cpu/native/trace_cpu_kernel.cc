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
#include "kernel/cpu/native/trace_cpu_kernel.h"
#include <type_traits>

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace trace_cpu {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kInputDim = 2;
constexpr size_t kOutputNum = 1;
}  // namespace

bool TraceCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  values_type_ = inputs.at(kIndex0)->dtype_id();
  return true;
}

int TraceCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = Convert2SizeT(inputs.at(kIndex0)->GetDeviceShapeVector());
  if (input_shape_.size() != kInputDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input tensor's dimension should be " << kInputDim
                      << ", but got " << input_shape_.size();
  }
  return KRET_OK;
}

bool TraceCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &,
                               const std::vector<kernel::KernelTensor *> &outputs) {
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
void TraceCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  size_t min_size = std::min(input_shape_[0], input_shape_[1]);
  const size_t stride = input_shape_[1];
  if constexpr (std::is_same_v<T, float16>) {
    float sum = 0.0f;
    for (size_t i = 0; i < min_size; ++i) {
      sum += static_cast<float>(*(input_addr + i * stride + i));
    }
    *output_addr = static_cast<float16>(sum);
  } else {
    T sum = static_cast<T>(0);
    for (size_t i = 0; i < min_size; ++i) {
      sum += *(input_addr + i * stride + i);
    }
    *output_addr = sum;
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Trace, TraceCpuKernelMod);
}  // namespace trace_cpu
}  // namespace kernel
}  // namespace mindspore
