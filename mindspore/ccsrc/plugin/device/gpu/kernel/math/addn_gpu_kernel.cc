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
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int AddNFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  workspace_size_list_.push_back(input_size_list_.front());
  return KRET_OK;
}

template <typename T>
bool AddNFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs) {
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  auto work_addr = output_addr;
  for (size_t i = 0; i < num_input_; i++) {
    if (output_addr == GetDeviceAddress<T>(inputs, i)) {
      work_addr = GetDeviceAddress<T>(workspace, 0);
      break;
    }
  }
  FillDeviceArray(outputs[0]->size / sizeof(T), output_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr_));
  FillDeviceArray(outputs[0]->size / sizeof(T), work_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr_));
  for (size_t i = 0; i < num_input_; i++) {
    T *input_addr = GetDeviceAddress<T>(inputs, i);
    if constexpr (std::is_same<T, Complex<float>>::value || std::is_same<T, Complex<double>>::value) {
      ElewiseComplexArith(outputs[0]->size / sizeof(T), BinaryOpType::kAdd, input_addr, work_addr, work_addr,
                          reinterpret_cast<cudaStream_t>(stream_ptr_));
    } else {
      ElewiseArith(outputs[0]->size / sizeof(T), BinaryOpType::kAdd, input_addr, work_addr, work_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr_));
    }
  }
  if (work_addr != output_addr) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_addr, work_addr, outputs[0]->size, cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "Addn cudaMemcpyAsync outputs failed");
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
