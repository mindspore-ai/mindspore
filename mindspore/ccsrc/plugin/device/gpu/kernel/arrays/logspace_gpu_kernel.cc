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
#include "plugin/device/gpu/kernel/arrays/logspace_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/logspace_impl.cuh"
namespace mindspore {
namespace kernel {
bool LogSpaceGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::LogSpace>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  // inputs and outputs should not be empty
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [ float16, float32, float64 ], "
                  << "but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  steps_ = kernel_ptr_->get_steps();
  base_ = kernel_ptr_->get_base();
  if (steps_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'steps' should be larger than 0, "
                  << "but got " << steps_;
    return false;
  }
  {
    size_t input_size = 2 * unit_size_;
    input_size_list_.emplace_back(input_size);
    size_t output_size = steps_ * unit_size_;
    output_size_list_.emplace_back(output_size);
  }
  return true;
}
int LogSpaceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  size_t input_size = 2 * unit_size_;
  input_size_list_.emplace_back(input_size);
  output_size_list_.emplace_back(steps_ * unit_size_);
  return KRET_OK;
}
void LogSpaceGpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
}
template <typename T>
bool LogSpaceGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (steps_ == 0) return true;
  auto start = GetDeviceAddress<T>(inputs, kIndex0);
  auto end = GetDeviceAddress<T>(inputs, kIndex1);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto status =
    CalLogSpace(start, end, steps_, base_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

// fp16, float, double
std::vector<std::pair<KernelAttr, LogSpaceGpuKernelMod::LogSpaceFunc>> LogSpaceGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &LogSpaceGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &LogSpaceGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &LogSpaceGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LogSpaceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, LogSpaceFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogSpace, LogSpaceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
