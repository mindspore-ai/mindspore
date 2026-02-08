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
#include "kernel/gpu/cuda/math/nextafter_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/nextafter_impl.cuh"

namespace mindspore {
namespace kernel {
bool NextAfterGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "' the kernel type should be in [ float32, float64], but got: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  int const INPUT_SIZE = 2;
  int const OUTPUT_SIZE = 1;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != INPUT_SIZE || outputs.size() != OUTPUT_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the operator should have 2 inputs and 1 outputs, but got "
                  << inputs.size() << "input(s) and " << outputs.size() << "output(s)";
    return false;
  }
  std::vector<size_t> input_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                         inputs.at(kIndex0)->GetDeviceShapeVector().end());
  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
  is_null_input_ = (input_elements_ == 0);
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }
  InitSizeLists();
  if (!is_input_dynamic_shape_.has_value()) {
    bool is_input_dynamic_shape = false;
    for (const auto &input : inputs) {
      auto input_shape = input->GetShapeVector();
      if (std::any_of(input_shape.begin(), input_shape.end(), [](int64_t dim) { return dim < 0; })) {
        is_input_dynamic_shape = true;
        break;
      }
    }
    is_input_dynamic_shape_ = is_input_dynamic_shape;
  }
  return true;
}

int NextAfterGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  if (is_input_dynamic_shape_.has_value() && is_input_dynamic_shape_.value()) {
    DestroyResource();
    ResetResource();
    if (!Init(inputs, outputs)) {
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

template <typename T>
bool NextAfterGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  T *input1 = GetDeviceAddress<T>(inputs, 0);
  T *input2 = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  MS_EXCEPTION_IF_NULL(input1);
  MS_EXCEPTION_IF_NULL(input2);
  MS_EXCEPTION_IF_NULL(output);
  auto status =
    NextAfter(input_elements_, input1, input2, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, NextAfterGpuKernelMod::NextAfterFunc>> NextAfterGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &NextAfterGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &NextAfterGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> NextAfterGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NextAfterFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NextAfter, NextAfterGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
