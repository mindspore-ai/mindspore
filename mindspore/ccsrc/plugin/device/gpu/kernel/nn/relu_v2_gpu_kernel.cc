/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/relu_v2_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/relu_impl.cuh"

namespace mindspore {
namespace kernel {
bool ReluV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 2;
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
  return true;
}

int ReluV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs[kIndex0]->GetDeviceShapeAdaptively());
  element_num_ = std::accumulate(x_shape.begin(), x_shape.end(), size_t(1), std::multiplies<>());
  output_size_list_.pop_back();
  auto mask_size = (element_num_ + 31) / 32 * sizeof(uint32_t);
  output_size_list_.push_back(mask_size);
  return KRET_OK;
}

template <typename T>
bool ReluV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto x = GetDeviceAddress<T>(inputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(x, false);
  auto y = GetDeviceAddress<T>(outputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(y, false);
  auto mask = GetDeviceAddress<uint32_t>(outputs, 1);
  MS_ERROR_IF_NULL_W_RET_VAL(mask, false);
  ReluV2(element_num_, x, y, mask, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, ReluV2GpuKernelMod::ReLUV2FwLaunchFunc>> ReluV2GpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt32),
   &ReluV2GpuKernelMod::LaunchKernel<uint8_t>},
};

std::vector<KernelAttr> ReluV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ReluV2GpuKernelMod::ReLUV2FwLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ReLUV2, ReluV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
