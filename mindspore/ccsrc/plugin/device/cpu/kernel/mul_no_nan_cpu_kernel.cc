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

#include "plugin/device/cpu/kernel/mul_no_nan_cpu_kernel.h"

#include <functional>
#include <vector>
#include <complex>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMulNoNanInputsNum = 2;
constexpr size_t kMulNoNanOutputsNum = 1;
constexpr size_t kNumber2 = 2;
}  // namespace

bool MulNoNanCPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMulNoNanInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMulNoNanOutputsNum, kernel_name_);

  input_dtype_ = inputs[kIndex0]->GetDtype();
  output_dtype_ = outputs[kIndex0]->GetDtype();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MulNoNanCPUKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input0_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input1_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, MulNoNanCPUKernelMod::KernelRunFunc>> &MulNoNanCPUKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, MulNoNanCPUKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &MulNoNanCPUKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &MulNoNanCPUKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &MulNoNanCPUKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &MulNoNanCPUKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &MulNoNanCPUKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &MulNoNanCPUKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &MulNoNanCPUKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &MulNoNanCPUKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &MulNoNanCPUKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MulNoNanCPUKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MulNoNanCPUKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &MulNoNanCPUKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &MulNoNanCPUKernelMod::LaunchKernel<std::complex<double>>},
  };
  return func_list;
}

template <typename T>
void MulNoNanCPUKernelMod::NoBcastCompute(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) {
  T *input_addr_0 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_addr_1 = static_cast<T *>(inputs[1]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t in0_elements_nums = inputs[0]->size / sizeof(T);
  size_t in1_elements_nums = inputs[1]->size / sizeof(T);
  size_t out_size = outputs[0]->size / sizeof(T);
  size_t type = in0_elements_nums == in1_elements_nums ? 0 : (in0_elements_nums == 1 ? 1 : kNumber2);

  auto task = [output_addr, input_addr_0, input_addr_1, type](size_t start, size_t end) {
    switch (type) {
      case 0:
        for (size_t i = start; i < end; ++i) {
          if (*(input_addr_1 + i) == static_cast<T>(0)) {
            *(output_addr + i) = static_cast<T>(0);
          } else {
            *(output_addr + i) = static_cast<T>(*(input_addr_0 + i) * *(input_addr_1 + i));
          }
        }
        break;
      case 1:
        for (size_t i = start; i < end; ++i) {
          if (*(input_addr_1 + i) == static_cast<T>(0)) {
            *(output_addr + i) = static_cast<T>(0);
          } else {
            *(output_addr + i) = *input_addr_0 * *(input_addr_1 + i);
          }
        }
        break;
      case 2:
        if (*input_addr_1 == static_cast<T>(0)) {
          for (size_t i = start; i < end; ++i) {
            *(output_addr + i) = static_cast<T>(0);
          }
        } else {
          for (size_t i = start; i < end; ++i) {
            *(output_addr + i) = static_cast<T>(*(input_addr_0 + i) * *input_addr_1);
          }
        }
        break;
      default:
        MS_LOG(EXCEPTION) << "Invalid type ";
    }
  };
  ParallelLaunchAutoSearch(task, out_size, this, &parallel_search_info_);
}

template <typename T>
void MulNoNanCPUKernelMod::BcastCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  T *input_addr_0 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_addr_1 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t out_size = outputs[0]->size / sizeof(T);
  BroadcastIterator base_iter(input0_shape_, input1_shape_, output_shape_);
  auto task = [&base_iter, output_addr, input_addr_0, input_addr_1](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; ++i) {
      if (input_addr_1[iter.GetInputPosB()] == static_cast<T>(0)) {
        output_addr[i] = static_cast<T>(0);
      } else {
        output_addr[i] = input_addr_0[iter.GetInputPosA()] * input_addr_1[iter.GetInputPosB()];
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, out_size, this, &parallel_search_info_);
}

template <typename T>
bool MulNoNanCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  size_t input0_elements_nums = inputs[0]->size / sizeof(T);
  size_t input1_elements_nums = inputs[1]->size / sizeof(T);
  bool no_bcast = (input0_shape_ == input1_shape_) || (input0_elements_nums == 1) || (input1_elements_nums == 1);
  if (no_bcast) {
    NoBcastCompute<T>(inputs, outputs);
  } else {
    BcastCompute<T>(inputs, outputs);
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MulNoNan, MulNoNanCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
