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
#include "plugin/device/gpu/kernel/random/randperm_gpu_kernel.h"
#include "mindspore/core/ops/randperm.h"

namespace mindspore {
namespace kernel {
bool RandpermGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  auto randperm_ptr = std::dynamic_pointer_cast<ops::Randperm>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(randperm_ptr, false);
  max_length_ = static_cast<size_t>(randperm_ptr->get_max_length());
  pad_ = randperm_ptr->get_pad();
  return true;
}

int RandpermGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool RandpermGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_device = GetDeviceAddress<T>(inputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(input_device, false);
  T *output_device = GetDeviceAddress<T>(outputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(output_device, false);

  int32_t n = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&n, input_device, sizeof(int32_t), cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     kernel_name_ + " Failed to copy error code to host.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(),
                                     kernel_name_ + " cudaDeviceSyncFailed in RandpermGpuKernelMod");

  if (static_cast<size_t>(n) > max_length_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', n (" << n << ") cannot exceed max_length_ (" << max_length_
                      << ")";
  }

  // might not be a significant performance gain if this kernel is executed in cuda,
  // so we do the calculations on host and copy to device afterwards.
  std::vector<T> output_host(max_length_);
  std::iota(output_host.begin(), output_host.begin() + n, 0);
  std::fill(output_host.begin() + n, output_host.end(), static_cast<T>(pad_));
  std::shuffle(output_host.begin(), output_host.begin() + n, rng_);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_device, &output_host[0], max_length_ * sizeof(T), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    kernel_name_ + " cudaMemcpyAsync for output_host failed");

  return true;
}

std::vector<std::pair<KernelAttr, RandpermGpuKernelMod::RandpermGpuLaunchFunc>> RandpermGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &RandpermGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &RandpermGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &RandpermGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &RandpermGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &RandpermGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &RandpermGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &RandpermGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &RandpermGpuKernelMod::LaunchKernel<uint64_t>},
};

std::vector<KernelAttr> RandpermGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, RandpermGpuKernelMod::RandpermGpuLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Randperm, RandpermGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
