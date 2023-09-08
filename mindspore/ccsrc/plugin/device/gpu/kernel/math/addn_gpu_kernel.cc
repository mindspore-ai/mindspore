/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/addn_gpu_kernel.h"

#include "mindspore/core/ops/math_ops.h"
namespace mindspore {
namespace kernel {
bool AddNFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = prim->name();
  num_input_ = GetValue<int64_t>(prim->GetAttr("n"));
  if (num_input_ != inputs.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be  " << num_input_ << ", but got "
                  << inputs.size();
    return false;
  }
  constexpr size_t output_num = 1;
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  for (const auto &input : inputs) {
    const auto &input_shape = input->GetShapeVector();
    if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](ShapeValueDType shape) { return (shape == 0); })) {
      empty_tensor_input_ = true;
      return true;
    }
  }
  empty_tensor_input_ = false;
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int AddNFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool AddNFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs) {
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  cudaError_t status = cudaErrorNotReady;
  status =
    FillDeviceArray(outputs[0]->size / sizeof(T), output_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  std::vector<int64_t> ele_shape = {static_cast<int64_t>(outputs[0]->size / sizeof(T))};
  for (size_t i = 0; i < num_input_; i++) {
    T *input_addr = GetDeviceAddress<T>(inputs, i);
    status = BinaryOpWithBroadcastCudaFunc<BinaryOpType::kAdd, T, T, T>(
      false, ele_shape, ele_shape, ele_shape, input_addr, output_addr, output_addr, device_id_,
      reinterpret_cast<cudaStream_t>(stream_ptr_));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, AddNFwdGpuKernelMod::KernelRunFunc>> &AddNFwdGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, AddNFwdGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &AddNFwdGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &AddNFwdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &AddNFwdGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &AddNFwdGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &AddNFwdGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &AddNFwdGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &AddNFwdGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &AddNFwdGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &AddNFwdGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &AddNFwdGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &AddNFwdGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &AddNFwdGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &AddNFwdGpuKernelMod::LaunchKernel<Complex<double>>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AddN, AddNFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
