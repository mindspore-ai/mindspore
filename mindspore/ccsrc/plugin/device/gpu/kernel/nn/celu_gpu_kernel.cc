/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/celu_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/celu_impl.cuh"

namespace mindspore {
namespace kernel {
bool CeluGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CeLU>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' cast celu ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  alpha_ = static_cast<double>(kernel_ptr->get_alpha());

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int CeluGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 1.";
    return KRET_RESIZE_FAILED;
  }
  input_elements_ = input_size_list_[0] / unit_size_;

  return KRET_OK;
}

template <typename T>
bool CeluGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  CalculateCelu(input, input_elements_, alpha_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

const std::vector<std::pair<KernelAttr, CeluGpuKernelMod::KernelRunFunc>> &CeluGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, CeluGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CeluGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CeluGpuKernelMod::LaunchKernel<float>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CeLU, CeluGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
