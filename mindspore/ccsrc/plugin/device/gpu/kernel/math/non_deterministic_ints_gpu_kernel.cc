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
#include <algorithm>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/math/non_deterministic_ints_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/non_deterministic_ints_impl.cuh"
namespace mindspore {
namespace kernel {
const int32_t kNumint32 = 4;
const int32_t kNumint64 = 2;
const int32_t kSizeint32 = 4;
const int32_t kSizeint64 = 8;

bool NonDeterministicIntsGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [Int32, Int64, UInt32, UInt64]"
                     ", but got: "
                  << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }

  return true;
}

bool NonDeterministicIntsGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *cuda_stream) {
  kernel_func_(this, inputs, workspace, outputs, cuda_stream);
  return true;
}

template <typename T>
bool NonDeterministicIntsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &workspace,
                                                    const std::vector<AddressPtr> &outputs, void *cuda_stream) {
  T *output = GetDeviceAddress<T>(outputs, 0);
  curandStatePhilox4_32_10_t *devStates = nullptr;
  void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
  devStates = reinterpret_cast<curandStatePhilox4_32_10_t *>(workspace_addr);
  LaunchNonDeterministicInts(devStates, output, output_num_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
  return true;
}

int NonDeterministicIntsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  ResetResource();

  std::vector<int64_t> input_shape_ = inputs[0]->GetShapeVector();
  std::vector<int64_t> output_shape_ = outputs[0]->GetShapeVector();

  input_num_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<int64_t>());
  output_num_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int64_t>());
  input_size_list_.emplace_back(input_num_ * unit_input_size_);
  output_size_list_.emplace_back(output_num_ * unit_output_size_);
  if (unit_output_size_ == kSizeint32) {
    // int32 or uint32.
    workspace_size_list_.emplace_back(output_num_ / kNumint32 * sizeof(curandStatePhilox4_32_10_t));
  } else if (unit_output_size_ == kSizeint64) {
    // int64 or uint64.
    workspace_size_list_.emplace_back(output_num_ / kNumint64 * sizeof(curandStatePhilox4_32_10_t));
  }
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, NonDeterministicIntsGpuKernelMod::KernelFunc>>
  NonDeterministicIntsGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt32),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &NonDeterministicIntsGpuKernelMod::LaunchKernel<uint64_t>}};

std::vector<KernelAttr> NonDeterministicIntsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NonDeterministicInts, NonDeterministicIntsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
