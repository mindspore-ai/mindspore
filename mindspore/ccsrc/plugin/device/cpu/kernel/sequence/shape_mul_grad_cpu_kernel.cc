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

#include "plugin/device/cpu/kernel/sequence/shape_mul_grad_cpu_kernel.h"
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
bool ShapeMulGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ShapeMulGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  auto dout_shape = inputs.at(kIndex1)->GetShapeVector();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  if (input_shape_.size() != 1 || !dout_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input_shape size must be 1, but got " << input_shapes_;
  }
  if (!IsSameShape(input_shape_, output_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << " the input and output shape must be equal, but input_shape = " << input_shape_
                      << " output shape = " << output_shape;
  }
  return KRET_OK;
}

template <typename T>
bool ShapeMulGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  int64_t *input_addr = GetDeviceAddress<int64_t>(inputs, 0);
  int64_t *dout_addr = GetDeviceAddress<int64_t>(inputs, 1);
  int64_t *output_addr = GetDeviceAddress<int64_t>(outputs, 0);
  int64_t out = 1;
  for (int64_t i = 0; i < input_shape_[0]; i++) {
    out *= input_addr[i];
  }
  for (int64_t i = 0; i < input_shape_[0]; i++) {
    output_addr[i] = *dout_addr * out / input_addr[i];
  }

  return true;
}  // namespace kernel

const std::vector<std::pair<KernelAttr, ShapeMulGradCpuKernelMod::KernelRunFunc>>
  &ShapeMulGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ShapeMulGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeMulGradCpuKernelMod::LaunchKernel<int64_t>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ShapeMulGrad, ShapeMulGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
