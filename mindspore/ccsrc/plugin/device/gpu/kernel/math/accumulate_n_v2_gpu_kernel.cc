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

#include "plugin/device/gpu/kernel/math/accumulate_n_v2_gpu_kernel.h"
#include <utility>
namespace mindspore {
namespace kernel {
bool AccumulateNV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::AccumulateNV2>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [half, float, double, int32, int8, uint8], "
                  << "but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  return true;
}

int AccumulateNV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  for (const auto &output : outputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto output_shape = output->GetShapeVector();
    if (!IsValidShape(output_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  n_ = inputs.size();
  size_t output_size = output_elements_ * unit_output_size_;
  input_size_list_.push_back(n_ * output_size);
  output_size_list_.push_back(output_size);
  workspace_size_list_.push_back(n_ * sizeof(void *));
  return KRET_OK;
}

template <typename T>
bool AccumulateNV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  T *output = GetDeviceAddress<T>(outputs, 0);
  T **inputs_array = GetDeviceAddress<T *>(workspace, 0);
  std::unique_ptr<T *[]> inputs_host = std::make_unique<T *[]>(n_);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs_host[i] = GetDeviceAddress<T>(inputs, i);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(inputs_array, inputs_host.get(), n_ * sizeof(T *), cudaMemcpyHostToDevice, stream),
    "cudaMemcpy failed.");
  auto status = CalAccumulateNV2(output_elements_, n_, inputs_array, output, device_id_, stream);
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, AccumulateNV2GpuKernelMod::ANVFunc>> AccumulateNV2GpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &AccumulateNV2GpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &AccumulateNV2GpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &AccumulateNV2GpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &AccumulateNV2GpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &AccumulateNV2GpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &AccumulateNV2GpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> AccumulateNV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ANVFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AccumulateNV2, AccumulateNV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
