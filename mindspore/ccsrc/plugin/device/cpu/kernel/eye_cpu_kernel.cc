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

#include "plugin/device/cpu/kernel/eye_cpu_kernel.h"
#include <algorithm>
#include <memory>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/eye.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kEyeInputsNum = 3;
constexpr size_t kEyeOutputsNum = 1;
}  // namespace
bool EyeCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::make_shared<ops::Eye>(base_operator->GetPrim());
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Eye ops failed!";
    return false;
  }
  num_n_ = kernel_ptr->get_num_rows();
  num_m_ = kernel_ptr->get_num_columns();
  if (num_n_ < 1) {
    MS_EXCEPTION(ValueError) << "For Eye, n is " << num_n_ << ", but n should be greater than 0.";
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int EyeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs,
                            const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  return 0;
}

template <typename T>
bool EyeCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  int64_t num_min = (num_n_ > num_m_) ? num_m_ : num_n_;
  size_t data_num = outputs[0]->size;
  size_t data_size = data_num * sizeof(T);
  auto ouput_ptr = outputs[0]->addr;
  (void)memset(ouput_ptr, 0, data_size);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  T num = static_cast<T>(1);
  for (int64_t i = 0; i < num_min; i++) {
    *(output_addr + (num_m_ + 1) * i) = static_cast<T>(num);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, EyeCpuKernelMod::KernelRunFunc>> &EyeCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, EyeCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddOutputAttr(kNumberTypeFloat16), &EyeCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddOutputAttr(kNumberTypeFloat32), &EyeCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddOutputAttr(kNumberTypeFloat64), &EyeCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddOutputAttr(kNumberTypeInt8), &EyeCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeInt16), &EyeCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeInt32), &EyeCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeInt64), &EyeCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeUInt8), &EyeCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeUInt16), &EyeCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeUInt32), &EyeCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeUInt64), &EyeCpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddOutputAttr(kNumberTypeComplex64), &EyeCpuKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr().AddOutputAttr(kNumberTypeComplex128), &EyeCpuKernelMod::LaunchKernel<std::complex<double>>},
    {KernelAttr().AddOutputAttr(kNumberTypeBool), &EyeCpuKernelMod::LaunchKernel<bool>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Eye, EyeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
