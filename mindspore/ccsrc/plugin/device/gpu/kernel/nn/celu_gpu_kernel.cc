/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
bool CeluGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
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

int CeluGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != 2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 1.";
    return KRET_RESIZE_FAILED;
  }
  input_elements_ = input_size_list_[0] / unit_size_;

  return KRET_OK;
}

template <typename T>
bool CeluGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                    const std::vector<KernelTensor *> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  double *alpha = GetDeviceAddress<double>(inputs, kIndex1);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto status =
    CalculateCelu(input, input_elements_, alpha, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, this->kernel_name_);
  return true;
}

const std::vector<std::pair<KernelAttr, CeluGpuKernelMod::KernelRunFunc>> &CeluGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, CeluGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat16),
     &CeluGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
     &CeluGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CeluGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CeLU, CeluGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
