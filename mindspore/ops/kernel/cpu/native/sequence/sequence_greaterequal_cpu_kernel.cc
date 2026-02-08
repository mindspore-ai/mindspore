/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/sequence/sequence_greaterequal_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool SequenceGreaterEqualCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SequenceGreaterEqualCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs[0]->GetShapeVector().empty() || inputs[1]->GetShapeVector().empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the x and y shape can't be 0, but got " << GetShapes(inputs);
  }
  x_size_ = static_cast<size_t>(inputs[0]->GetShapeVector()[0]);
  y_size_ = static_cast<size_t>(inputs[1]->GetShapeVector()[0]);
  return KRET_OK;
}

template <typename T, typename S>
bool SequenceGreaterEqualCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &,
                                                    const std::vector<KernelTensor *> &outputs) {
  const auto input_0_addr = GetDeviceAddress<T>(inputs, 0);
  const auto input_1_addr = GetDeviceAddress<S>(inputs, 1);
  bool *output_addr = GetDeviceAddress<bool>(outputs, 0);
  MS_EXCEPTION_IF_NULL(input_0_addr);
  MS_EXCEPTION_IF_NULL(input_1_addr);

  size_t len = std::max(x_size_, y_size_);
  bool flag = true;
  for (size_t i = 0; i < len; i++) {
    if (i >= x_size_) {
      flag = false;
      break;
    }
    if (i >= y_size_) {
      flag = true;
      break;
    }
    if (static_cast<double>(input_0_addr[i]) < static_cast<double>(input_1_addr[i])) {
      flag = false;
      break;
    }
    if (static_cast<double>(input_0_addr[i]) > static_cast<double>(input_1_addr[i])) {
      flag = true;
      break;
    }
  }
  if (output_addr != nullptr) {
    output_addr[0] = flag;
    return true;
  }
  return false;
}

#define ADD_KERNEL(x_dtype, y_dtype, x_type, y_type)                  \
  {                                                                   \
    KernelAttr()                                                      \
      .AddInputAttr(kObjectTypeTuple, kNumberType##x_dtype)           \
      .AddInputAttr(kObjectTypeTuple, kNumberType##y_dtype)           \
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),             \
      &SequenceGreaterEqualCpuKernelMod::LaunchKernel<x_type, y_type> \
  }

const std::vector<std::pair<KernelAttr, SequenceGreaterEqualCpuKernelMod::KernelRunFunc>>
  &SequenceGreaterEqualCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SequenceGreaterEqualCpuKernelMod::KernelRunFunc>> func_list = {
    ADD_KERNEL(Float32, Float32, float, float), ADD_KERNEL(Float32, Float64, float, double),
    ADD_KERNEL(Float32, Int32, float, int),     ADD_KERNEL(Float32, Int64, float, int64_t),
    ADD_KERNEL(Float32, Bool, float, bool),     ADD_KERNEL(Float64, Float32, double, float),
    ADD_KERNEL(Float64, Bool, double, bool),    ADD_KERNEL(Float64, Float64, double, double),
    ADD_KERNEL(Float64, Int32, double, int),    ADD_KERNEL(Float64, Int64, double, int64_t),
    ADD_KERNEL(Int32, Float32, int, float),     ADD_KERNEL(Int32, Float64, int, double),
    ADD_KERNEL(Int32, Int32, int, int),         ADD_KERNEL(Int32, Int64, int, int64_t),
    ADD_KERNEL(Int32, Bool, int, bool),         ADD_KERNEL(Int64, Float32, int64_t, float),
    ADD_KERNEL(Int64, Bool, int64_t, bool),     ADD_KERNEL(Int64, Float64, int64_t, double),
    ADD_KERNEL(Int64, Int32, int64_t, int),     ADD_KERNEL(Int64, Int64, int64_t, int64_t),
    ADD_KERNEL(Bool, Int32, bool, int),         ADD_KERNEL(Bool, Int64, bool, int64_t),
    ADD_KERNEL(Bool, Bool, bool, bool),         ADD_KERNEL(Bool, Float64, bool, double),
    ADD_KERNEL(Bool, Float32, bool, float)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, tuple_greater_equal, SequenceGreaterEqualCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, list_greater_equal, SequenceGreaterEqualCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
