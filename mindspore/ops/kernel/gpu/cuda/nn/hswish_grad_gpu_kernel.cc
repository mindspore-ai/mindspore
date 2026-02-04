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

#include "kernel/gpu/cuda/nn/hswish_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/hswish_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHSwishGradInputsNum = 2;
constexpr size_t kHSwishGradOutputsNum = 1;
}  // namespace

bool HSwishGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHSwishGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHSwishGradOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int HSwishGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "y_grad") ||
                   CHECK_SHAPE_NULL(inputs[kIndex1]->GetShapeVector(), kernel_name_, "x");
  input_size_ = SizeOf(input_shape);
  return KRET_OK;
}

template <typename T>
bool HSwishGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  auto *input = GetDeviceAddress<T>(inputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto *x = GetDeviceAddress<T>(inputs, 1);
  MS_ERROR_IF_NULL_W_RET_VAL(x, false);
  auto *output = GetDeviceAddress<T>(outputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);
  auto status = CalHSwishGrad(input_size_, input, x, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, HSwishGradGpuKernelMod::HSwishGradLaunchFunc>> HSwishGradGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &HSwishGradGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &HSwishGradGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &HSwishGradGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &HSwishGradGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &HSwishGradGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &HSwishGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &HSwishGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> HSwishGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, HSwishGradGpuKernelMod::HSwishGradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, HSwishGrad, HSwishGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
