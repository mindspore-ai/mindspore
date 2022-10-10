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
#define MULNONAN_COMPUTE_CASE(DTYPE, TYPE, INPUTS, OUTPUTS) \
  case (DTYPE): {                                           \
    LaunchKernel<TYPE>(INPUTS, OUTPUTS);                    \
    break;                                                  \
  }
}  // namespace

void MulNoNanCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  output_dtype_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  input0_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input1_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
}

bool MulNoNanCPUKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMulNoNanInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMulNoNanOutputsNum, kernel_name_);
  switch (input_dtype_) {
    MULNONAN_COMPUTE_CASE(kNumberTypeInt8, int8_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeInt16, int16_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeInt32, int32_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeInt64, int64_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeUInt8, uint8_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeUInt16, uint16_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeUInt32, uint32_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeUInt64, uint64_t, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeFloat16, float16, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeFloat32, float, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeFloat64, double, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeComplex64, std::complex<float>, inputs, outputs)
    MULNONAN_COMPUTE_CASE(kNumberTypeComplex128, std::complex<double>, inputs, outputs)
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input 'x' "
                        << TypeIdToType(input_dtype_)->ToString() << " not support.";
  }
  return true;
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
void MulNoNanCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  size_t input0_elements_nums = inputs[0]->size / sizeof(T);
  size_t input1_elements_nums = inputs[1]->size / sizeof(T);
  bool no_bcast = (input0_shape_ == input1_shape_) || (input0_elements_nums == 1) || (input1_elements_nums == 1);
  if (no_bcast) {
    NoBcastCompute<T>(inputs, outputs);
  } else {
    BcastCompute<T>(inputs, outputs);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MulNoNan, MulNoNanCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
