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

#include "kernel/gpu/cuda/nn/gelu_grad_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/gelu_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGeLUGradInputsNum = 3;
constexpr size_t kGeLUGradOutputsNum = 1;
}  // namespace

bool GeLUGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGeLUGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGeLUGradOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GeLUGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "dy") ||
                   CHECK_SHAPE_NULL(inputs[kIndex1]->GetShapeVector(), kernel_name_, "x") ||
                   CHECK_SHAPE_NULL(inputs[kIndex2]->GetShapeVector(), kernel_name_, "y");
  input_size_ = SizeOf(input_shape);
  return KRET_OK;
}

template <typename T>
bool GeLUGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *dy_addr = GetDeviceAddress<T>(inputs, 0);
  T *x_addr = GetDeviceAddress<T>(inputs, 1);
  T *dx_addr = GetDeviceAddress<T>(outputs, 0);

  auto status = GeluGradKernel(input_size_, dy_addr, x_addr, dx_addr, reinterpret_cast<cudaStream_t>(stream_ptr),
                               GET_CTX_DEVICE_ID);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, GeLUGradGpuKernelMod::GeLUGradGpuLaunchFunc>> GeLUGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &GeLUGradGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GeLUGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GeLUGradGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> GeLUGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, GeLUGradGpuKernelMod::GeLUGradGpuLaunchFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GeLUGrad, GeLUGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
