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

#include "plugin/device/gpu/kernel/arrays/inplace_op_gpu_kernel.h"
#include <unordered_map>
#include <string>
namespace mindspore {
namespace kernel {
static std::unordered_map<std::string, int> op_type_map = {
  {"InplaceUpdate", INPLACE_OP_TYPE_UPDATE}, {"InplaceAdd", INPLACE_OP_TYPE_ADD}, {"InplaceSub", INPLACE_OP_TYPE_SUB}};
bool InplaceOpGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = op_type_map.find(kernel_name_);
  if (iter == op_type_map.end()) {
    MS_LOG(ERROR) << "For InplaceOp kernel, Can only support InplaceUpdate, InplaceAdd, InplaceSub, but got "
                  << kernel_name_;
    return false;
  }
  kernel_type_ = iter->second;
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
  unit_size_ = abstract::TypeIdSize(inputs[0]->GetDtype());
  if (kernel_name_ == "InplaceUpdate") {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::InplaceUpdate>(base_operator);
    indices_ = kernel_ptr->get_indices();
  } else if (kernel_name_ == "InplaceAdd") {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::InplaceAdd>(base_operator);
    indices_ = kernel_ptr->get_indices();
  } else {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::InplaceSub>(base_operator);
    indices_ = kernel_ptr->get_indices();
  }
  return true;
}

int InplaceOpGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  first_dimension_ = input_shape_x[kIndex0];
  input_elements_x = std::accumulate(input_shape_x.begin(), input_shape_x.end(), 1, std::multiplies<int64_t>());
  input_elements_v = std::accumulate(input_shape_v.begin(), input_shape_v.end(), 1, std::multiplies<int64_t>());
  size_t input_size_x = input_elements_x * unit_size_;
  size_t input_size_v = input_elements_v * unit_size_;
  size_t indices_size = indices_.size() * sizeof(int64_t);
  input_size_list_.push_back(input_size_x);
  input_size_list_.push_back(input_size_v);
  output_size_list_.push_back(input_size_x);
  workspace_size_list_.push_back(indices_size);
  if (kernel_name_ == "InplaceUpdate") {
    workspace_size_list_.push_back(indices_size);
  }
  return KRET_OK;
}

void InplaceOpGpuKernelMod::ResetResource() noexcept {
  band_size_ = 1;
  input_elements_x = 0;
  input_elements_v = 0;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool InplaceOpGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *input_x = GetDeviceAddress<T>(inputs, kIndex0);
  T *input_v = GetDeviceAddress<T>(inputs, kIndex1);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto indices_ptr = GetDeviceAddress<int64_t>(workspace, kIndex0);
  int64_t *indices_key_ptr = nullptr;
  if (kernel_name_ == "InplaceUpdate") {
    indices_key_ptr = GetDeviceAddress<int64_t>(workspace, kIndex1);
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);

  // Copy from 'x' into 'y'.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input_x, input_elements_x * unit_size_, cudaMemcpyDeviceToDevice, cuda_stream),
    "cudaMemcpyAsync output 'output' from 'input_x' failed.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(indices_ptr, indices_.data(), indices_.size() * sizeof(int64_t),
                                                     cudaMemcpyHostToDevice, cuda_stream),
                                     "cudaMemcpyAsync indices variable failed.");
  CalInplaceOp(input_elements_v, input_v, output, indices_ptr, indices_key_ptr, first_dimension_, band_size_,
               device_id_, kernel_type_, cuda_stream);
  return true;
}

std::vector<std::pair<KernelAttr, InplaceOpGpuKernelMod::InplaceOpFunc>> InplaceOpGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &InplaceOpGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &InplaceOpGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &InplaceOpGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &InplaceOpGpuKernelMod::LaunchKernel<int>}};

std::vector<KernelAttr> InplaceOpGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InplaceOpFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InplaceUpdate, InplaceOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InplaceAdd, InplaceOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InplaceSub, InplaceOpGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
