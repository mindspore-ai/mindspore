/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/inplace_op_v2_gpu_kernel.h"
#include <unordered_map>
#include <string>
namespace mindspore {
namespace kernel {
static std::unordered_map<std::string, int> op_type_map = {{"InplaceUpdateV2", INPLACE_OP_TYPE_UPDATE}};
bool InplaceOpV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = op_type_map.find(kernel_name_);
  if (iter == op_type_map.end()) {
    MS_LOG(ERROR) << "For InplaceOpV2 kernel, Can only support InplaceUpdateV2, but got " << kernel_name_;
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
  return true;
}

int InplaceOpV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  std::vector<int64_t> input_shape_indices = std::vector<int64_t>(
    inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(), inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> input_shape_v = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  band_size_ = 1;
  for (size_t i = 1; i < input_shape_x.size(); ++i) {
    band_size_ *= input_shape_x[i];
  }
  input_elements_x = std::accumulate(input_shape_x.begin(), input_shape_x.end(), 1, std::multiplies<int64_t>());
  input_elements_v = std::accumulate(input_shape_v.begin(), input_shape_v.end(), 1, std::multiplies<int64_t>());
  size_t input_size_x = input_elements_x * unit_size_;
  size_t indices_size = input_shape_indices.size() * sizeof(int32_t);
  size_t input_size_v = input_elements_v * unit_size_;
  input_size_list_.push_back(input_size_x);
  input_size_list_.push_back(IntToSize(indices_size));
  input_size_list_.push_back(input_size_v);
  output_size_list_.push_back(input_size_x);
  if (kernel_name_ == ops::kNameInplaceUpdateV2) {
    workspace_size_list_.push_back(indices_size);
  }
  return KRET_OK;
}

void InplaceOpV2GpuKernelMod::ResetResource() noexcept {
  band_size_ = 1;
  input_elements_x = 0;
  input_elements_v = 0;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool InplaceOpV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  T *input_x = GetDeviceAddress<T>(inputs, kIndex0);
  int32_t *input_indices = GetDeviceAddress<int32_t>(inputs, kIndex1);
  T *input_v = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  int32_t *indices_key_ptr = nullptr;
  if (kernel_name_ == ops::kNameInplaceUpdateV2) {
    indices_key_ptr = GetDeviceAddress<int32_t>(workspace, kIndex0);
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);

  // Copy from 'x' into 'y'.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input_x, input_elements_x * unit_size_, cudaMemcpyDeviceToDevice, cuda_stream),
    "cudaMemcpyAsync output 'output' from 'input_x' failed.");
  CalInplaceOp(input_elements_v, input_v, output, input_indices, indices_key_ptr, band_size_, device_id_, kernel_type_,
               cuda_stream);
  return true;
}

std::vector<std::pair<KernelAttr, InplaceOpV2GpuKernelMod::InplaceOpFunc>> InplaceOpV2GpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &InplaceOpV2GpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &InplaceOpV2GpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &InplaceOpV2GpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &InplaceOpV2GpuKernelMod::LaunchKernel<int>}};

std::vector<KernelAttr> InplaceOpV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InplaceOpFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, InplaceUpdateV2, InplaceOpV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
