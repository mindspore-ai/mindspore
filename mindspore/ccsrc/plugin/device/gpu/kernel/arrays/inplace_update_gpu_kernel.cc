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

#include "plugin/device/gpu/kernel/arrays/inplace_update_gpu_kernel.h"
namespace mindspore {
namespace kernel {
bool InplaceUpdateGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::InplaceUpdate>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float16, float32, float64, int32]"
                     ", but got: "
                  << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  indices_ = kernel_ptr_->get_indices();
  return true;
}

int InplaceUpdateGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  std::vector<int64_t> input_shape_x = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> input_shape_v = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  band_size_ = 1;
  for (size_t i = 1; i < input_shape_x.size(); ++i) {
    band_size_ *= input_shape_x[i];
  }
  input_elements_x = std::accumulate(input_shape_x.begin(), input_shape_x.end(), 1, std::multiplies<int64_t>());
  input_elements_v = std::accumulate(input_shape_v.begin(), input_shape_v.end(), 1, std::multiplies<int64_t>());
  size_t input_size_x = input_elements_x * unit_size_;
  size_t input_size_v = input_elements_v * unit_size_;
  size_t indices_size = indices_.size() * sizeof(int64_t);
  input_size_list_.push_back(input_size_x);
  input_size_list_.push_back(input_size_v);
  output_size_list_.push_back(input_size_x);
  workspace_size_list_.push_back(indices_size);
  return KRET_OK;
}

void InplaceUpdateGpuKernelMod::ResetResource() noexcept {
  band_size_ = 1;
  input_elements_x = 0;
  input_elements_v = 0;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool InplaceUpdateGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  T *input_x = GetDeviceAddress<T>(inputs, kIndex0);
  T *input_v = GetDeviceAddress<T>(inputs, kIndex1);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto indices_ptr = GetDeviceAddress<int64_t>(workspace, kIndex0);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);

  // Copy from 'x' into 'y'.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input_x, input_elements_x * unit_size_, cudaMemcpyDeviceToDevice, cuda_stream),
    "cudaMemcpyAsync output 'output' from 'input_x' failed.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(indices_ptr, indices_.data(), indices_.size() * sizeof(int64_t),
                                                     cudaMemcpyHostToDevice, cuda_stream),
                                     "cudaMemcpyAsync indices variable failed.");
  CalInplaceUpdate(input_elements_v, input_v, output, indices_ptr, band_size_, device_id_, cuda_stream);
  return true;
}

std::vector<std::pair<KernelAttr, InplaceUpdateGpuKernelMod::InplaceUpdateFunc>> InplaceUpdateGpuKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    &InplaceUpdateGpuKernelMod::LaunchKernel<half>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    &InplaceUpdateGpuKernelMod::LaunchKernel<float>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    &InplaceUpdateGpuKernelMod::LaunchKernel<double>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    &InplaceUpdateGpuKernelMod::LaunchKernel<int>}};

std::vector<KernelAttr> InplaceUpdateGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InplaceUpdateFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InplaceUpdate, InplaceUpdateGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
