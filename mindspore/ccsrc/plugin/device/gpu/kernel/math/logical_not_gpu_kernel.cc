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

#include "plugin/device/gpu/kernel/math/logical_not_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise_op_impl.cuh"

namespace mindspore {
namespace kernel {
bool LogicalNotGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int LogicalNotGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  input_num_ = std::accumulate(x_shape.begin(), x_shape.end(), size_t(1), std::multiplies<>());
  return KRET_OK;
}

template <typename T>
bool LogicalNotGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  auto input_addr = GetDeviceAddress<T>(inputs, 0);
  auto output_addr = GetDeviceAddress<bool>(outputs, 0);
  LogicalNot(input_addr, output_addr, static_cast<size_t>(input_num_), stream_ptr_);
  return true;
}

const std::vector<std::pair<KernelAttr, LogicalNotGpuKernelMod::KernelRunFunc>> &LogicalNotGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, LogicalNotGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &LogicalNotGpuKernelMod::LaunchKernel<bool>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogicalNot, LogicalNotGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
