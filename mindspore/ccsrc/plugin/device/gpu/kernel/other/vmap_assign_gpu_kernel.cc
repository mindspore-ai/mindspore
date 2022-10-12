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

#include "plugin/device/gpu/kernel/other/vmap_assign_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include <complex>
#include "mindspore/core/ops/vmap_assign.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pack.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unpack.cuh"

namespace mindspore {
namespace kernel {
bool VmapAssignGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (kernel_name_ != prim::kPrimVmapStackAssign->name() && kernel_name_ != prim::kPrimVmapUnstackAssign->name()) {
    MS_LOG(ERROR) << "For 'VmapAssignGpuKernelMod', it's must be VmapStackAssign or VmapUnstackAssign but get "
                  << "invalid kernel name : " << kernel_name_;
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int VmapAssignGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  size_t inputs_size = inputs.size();
  if (inputs_size < kInputLowerLimit) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's inputs size is invalid, must be large than "
                  << kInputLowerLimit << ", but got " << inputs_size << ".";
    return KRET_RESIZE_FAILED;
  } else {
    stack_num_ = inputs_size - 1;
  }

  auto input_shape = inputs.at(kIndex1)->GetShapeVector();
  dims_axis_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  (void)workspace_size_list_.emplace_back(stack_num_ * sizeof(void *));

  auto stacked_param_shape = inputs.at(kIndex0)->GetShapeVector();
  stacked_param_size_ =
    std::accumulate(stacked_param_shape.begin(), stacked_param_shape.end(), size_t(1), std::multiplies<size_t>());
  return KRET_OK;
}

template <typename T>
bool VmapAssignGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  T *stacked_param = GetDeviceAddress<T>(inputs, kIndex0);
  T **partition_array = GetDeviceAddress<T *>(workspace, kIndex0);
  auto partition_host = std::make_unique<T *[]>(stack_num_);
  for (size_t i = 0; i < inputs.size() - 1; i++) {
    partition_host[i] = GetDeviceAddress<T>(inputs, i + 1);
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(partition_array, partition_host.get(), sizeof(T *) * stack_num_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "VmapAssign opt cudaMemcpyAsync partition_array failed");

  if (kernel_name_ == prim::kPrimVmapStackAssign->name()) {
    PackKernel(stacked_param_size_, stack_num_, dims_axis_, partition_array, stacked_param,
               reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    UnpackKernel(stacked_param_size_, stack_num_, dims_axis_, partition_array, stacked_param,
                 reinterpret_cast<cudaStream_t>(cuda_stream_));
  }

  int *output_address = GetDeviceAddress<int>(outputs, kIndex0);
  int output = 1;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output_address, &output, sizeof(int), cudaMemcpyHostToDevice,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cudaMemcpyAsync output failed");
  return true;
}

const std::vector<std::pair<KernelAttr, VmapAssignGpuKernelMod::KernelRunFunc>> &VmapAssignGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, VmapAssignGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
     &VmapAssignGpuKernelMod::LaunchKernel<bool>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, VmapStackAssign, VmapAssignGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, VmapUnstackAssign, VmapAssignGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
