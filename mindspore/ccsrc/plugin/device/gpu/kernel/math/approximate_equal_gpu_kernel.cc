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

#include "plugin/device/gpu/kernel/math/approximate_equal_gpu_kernel.h"
namespace mindspore {
namespace kernel {
bool ApproximateEqualGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::ApproximateEqual>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float16, float32, float64]"
                     ", but got: "
                  << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);

  tolerance_ = kernel_ptr_->get_tolerance();
  return true;
}

int ApproximateEqualGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<int64_t> input_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
  size_t input_size = input_elements_ * unit_size_;
  input_size_list_.push_back(input_size);
  input_size_list_.push_back(input_size);
  output_size_list_.push_back(input_elements_ * sizeof(bool));
  return KRET_OK;
}

void ApproximateEqualGpuKernelMod::ResetResource() noexcept {
  input_elements_ = 0;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
}

template <typename T>
bool ApproximateEqualGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  T *input_x1 = GetDeviceAddress<T>(inputs, 0);
  T *input_x2 = GetDeviceAddress<T>(inputs, 1);
  bool *output = GetDeviceAddress<bool>(outputs, 0);
  auto status = CalApproximateEqual(input_elements_, input_x1, input_x2, tolerance_, output, device_id_,
                                    reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ApproximateEqualGpuKernelMod::ApproximateEqualFunc>>
  ApproximateEqualGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     &ApproximateEqualGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     &ApproximateEqualGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     &ApproximateEqualGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ApproximateEqualGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApproximateEqualFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApproximateEqual, ApproximateEqualGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
