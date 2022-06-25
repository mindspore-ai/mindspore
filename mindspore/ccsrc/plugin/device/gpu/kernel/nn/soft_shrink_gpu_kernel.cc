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

#include "plugin/device/gpu/kernel/nn/soft_shrink_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/soft_shrink_impl.cuh"
#include "mindspore/core/ops/soft_shrink.h"

namespace mindspore {
namespace kernel {
#define SOFT_SHRINK_GPU_REGISTER(DT, T) \
  KernelAttr().AddInputAttr(DT).AddOutputAttr(DT), &SoftShrinkGpuKernelMod::LaunchKernel<T>

template <typename T>
bool SoftShrinkGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  SoftShrink(size_, input_addr, lambd_, output_addr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

bool SoftShrinkGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::SoftShrink>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast SoftShrink ops failed!";
    return false;
  }
  lambd_ = kernel_ptr->get_lambd();

  if (auto ret = MatchKernelFunc(base_operator, inputs, outputs); !ret) {
    return ret;
  }
  return true;
}

int SoftShrinkGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  auto in_shape = inputs[kIndex0]->GetShapeVector();
  size_ = std::accumulate(in_shape.begin(), in_shape.end(), size_t(1), std::multiplies<size_t>());
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, SoftShrinkGpuKernelMod::KernelRunFunc>> &SoftShrinkGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, SoftShrinkGpuKernelMod::KernelRunFunc>> func_list = {
    {SOFT_SHRINK_GPU_REGISTER(kNumberTypeFloat32, float)},
    {SOFT_SHRINK_GPU_REGISTER(kNumberTypeFloat16, half)},
    {SOFT_SHRINK_GPU_REGISTER(kNumberTypeInt32, int32_t)},
    {SOFT_SHRINK_GPU_REGISTER(kNumberTypeInt64, int64_t)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SoftShrink, SoftShrinkGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
