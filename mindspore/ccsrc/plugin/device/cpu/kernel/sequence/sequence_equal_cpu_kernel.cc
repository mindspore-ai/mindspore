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

#include "plugin/device/cpu/kernel/sequence/sequence_equal_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool SequenceEqualCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int SequenceEqualCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  is_inputs_type_diff_ = false;
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(input_shapes_.size(), kInputsNum, kernel_name_);
  if (input_shapes_[0].empty() || input_shapes_[1].empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the x and y shape can't be 0, but got " << input_shapes_;
  }
  x_size_ = input_shapes_[0][0];
  y_size_ = input_shapes_[1][0];
  if (inputs[0]->GetDtype() != inputs[1]->GetDtype()) {
    is_inputs_type_diff_ = true;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool SequenceEqualCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs) {
  const auto x_addr = GetDeviceAddress<T>(inputs, 0);
  const auto y_addr = GetDeviceAddress<S>(inputs, 1);
  bool *output_addr = GetDeviceAddress<bool>(outputs, 0);
  if (x_size_ != y_size_ || is_inputs_type_diff_) {
    *output_addr = false;
    return true;
  }
  for (int64_t i = 0; i < x_size_; ++i) {
    if (static_cast<double>(x_addr[i]) != static_cast<double>(y_addr[i])) {
      *output_addr = false;
      return true;
    }
  }
  *output_addr = true;
  return true;
}

#define ADD_KERNEL(x_dtype, y_dtype, x_type, y_type)           \
  {                                                            \
    KernelAttr()                                               \
      .AddInputAttr(kObjectTypeTuple, kNumberType##x_dtype)    \
      .AddInputAttr(kObjectTypeTuple, kNumberType##y_dtype)    \
      .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool),      \
      &SequenceEqualCpuKernelMod::LaunchKernel<x_type, y_type> \
  }

const std::vector<std::pair<KernelAttr, SequenceEqualCpuKernelMod::KernelRunFunc>>
  &SequenceEqualCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SequenceEqualCpuKernelMod::KernelRunFunc>> func_list = {
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
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, tuple_equal, SequenceEqualCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
