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

#include "plugin/device/cpu/kernel/select_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include "mindspore/core/ops/select.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;
constexpr size_t kSelectInputsNum = 3;
constexpr size_t kSelectOutputsNum = 1;
}  // namespace

bool SelectCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Select>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  kernel_name_ = kernel_ptr->name();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int SelectCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape = inputs[0]->GetShapeVector();
  element_num_ = SizeOf(input_shape);
  return KRET_OK;
}

template <typename T>
bool SelectCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSelectInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSelectOutputsNum, kernel_name_);
  auto *input_cond = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *input_x = reinterpret_cast<T *>(inputs[1]->addr);
  auto *input_y = reinterpret_cast<T *>(inputs[2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  auto task = [&input_x, &input_y, &output, &input_cond](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = input_cond[i] ? input_x[i] : input_y[i];
    }
  };
  ParallelLaunchAutoSearch(task, element_num_, this, &parallel_search_info_);
  return true;
}

using selectPair = std::pair<KernelAttr, SelectCpuKernelMod::KernelRunFunc>;
const std::vector<selectPair> &SelectCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SelectCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SelectCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SelectCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SelectCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &SelectCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &SelectCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SelectCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SelectCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &SelectCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16),
     &SelectCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &SelectCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &SelectCpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool),
     &SelectCpuKernelMod::LaunchKernel<bool>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SelectCpuKernelMod::LaunchKernel<float_complex>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SelectCpuKernelMod::LaunchKernel<double_complex>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Select, SelectCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
