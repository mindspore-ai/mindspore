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

#include "plugin/device/gpu/kernel/arrays/check_numerics_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr size_t kNumber2 = 2;
bool CheckNumericsGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::CheckNumerics>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [float16, float32, float64], "
                  << "but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  return true;
}

int CheckNumericsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  size_t output_size = output_elements_ * unit_output_size_;
  input_size_list_.push_back(output_size);
  output_size_list_.push_back(output_size);
  workspace_size_list_.push_back(kNumber2 * sizeof(int32_t));
  return KRET_OK;
}

template <typename T>
bool CheckNumericsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  int32_t *flag_device = GetDeviceAddress<int32_t>(workspace, 0);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  int32_t flag_host[2];
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(flag_device, 0, kNumber2 * sizeof(int32_t), stream),
                                     "flag_check cudaMemsetAsync failed.");
  CalCheckNumerics(output_elements_, input, flag_device, device_id_, stream);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(flag_host, flag_device, kNumber2 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
    "flag_host cudaMemcpy failed.");
  if (flag_host[0] == 1 && flag_host[1] == 1) {
    MS_EXCEPTION(ValueError) << ": Tensor had Inf and NaN values [Op" << kernel_name_ << "].";
  } else if (flag_host[0] == 1 && flag_host[1] == 0) {
    MS_EXCEPTION(ValueError) << ": Tensor had NaN values [Op" << kernel_name_ << "].";
  } else if (flag_host[0] == 0 && flag_host[1] == 1) {
    MS_EXCEPTION(ValueError) << ": Tensor had Inf values [Op" << kernel_name_ << "].";
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output, input, output_elements_ * sizeof(T), cudaMemcpyDeviceToDevice, stream),
      "cudaMemcpyAsync value variable failed.");
  }
  return true;
}

std::vector<std::pair<KernelAttr, CheckNumericsGpuKernelMod::CNFunc>> CheckNumericsGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &CheckNumericsGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &CheckNumericsGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &CheckNumericsGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CheckNumericsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CNFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CheckNumerics, CheckNumericsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
