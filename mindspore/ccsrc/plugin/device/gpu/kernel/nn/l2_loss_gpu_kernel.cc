/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/l2_loss_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool L2LossGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int L2LossGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
  input_size_ = SizeOf(input_shape);
  if (input_size_ == 0) {
    return KRET_UNKNOWN_SHAPE;
  }
  return KRET_OK;
}

template <typename T>
bool L2LossGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  L2Loss(input_size_, input, output, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

const std::vector<std::pair<KernelAttr, L2LossGpuKernelMod::KernelRunFunc>> &L2LossGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, L2LossGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &L2LossGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &L2LossGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &L2LossGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, L2Loss, L2LossGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
