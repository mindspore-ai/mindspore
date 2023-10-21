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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/uniform_impl.cuh"
#include "plugin/device/gpu/kernel/random/uniform_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool UniformGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  kernel_func_ = func_list_[index].second;
  from_ = GetValue<float>(base_operator->GetAttr("from"));
  to_ = GetValue<float>(base_operator->GetAttr("to"));
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("seed")));
  uint64_t offset = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("offset")));
  seed_ = random::GetSeed(seed, offset);
  return true;
}

int UniformGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  input_size_ = SizeOf(inputs[0]->GetShapeVector());
  return KRET_OK;
}

template <typename T>
bool UniformGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, kIndex0);
  T *y = GetDeviceAddress<T>(outputs, kIndex0);
  auto status = CalUniform(x, y, from_, to_, seed_, seed_offset_, input_size_, device_id_,
                           reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  seed_offset_ += 1;
  return true;
}

std::vector<std::pair<KernelAttr, UniformGpuKernelMod::UniformFunc>> UniformGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &UniformGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &UniformGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &UniformGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> UniformGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UniformFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Uniform, UniformGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
