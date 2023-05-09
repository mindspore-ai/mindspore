/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unaddc_div required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/cpu/kernel/addcdiv_cpu_kernel.h"

#include <limits>
#include <utility>
#include <vector>
#include <cmath>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/arithmetic_cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/arithmetic_parameter.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/mul_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/power_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/sub_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
#define ADD_KERNEL(t1, t2, t3, t4, t5) \
  KernelAttr()                         \
    .AddInputAttr(kNumberType##t1)     \
    .AddInputAttr(kNumberType##t2)     \
    .AddInputAttr(kNumberType##t3)     \
    .AddInputAttr(kNumberType##t4)     \
    .AddOutputAttr(kNumberType##t5)
const int64_t kOutputNum = 1;
const int64_t kInputNum = 4;
const int64_t kInputData = 0;
const int64_t kInputX1 = 1;
const int64_t kInputX2 = 2;
const int64_t kInputValue = 3;
const int64_t kOutputData = 0;
}  // namespace

bool AddcdivCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  return true;
}

int AddcdivCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs[kInputData]->GetDtype();
  dtype_value = inputs[kInputValue]->GetDtype();
  input_shape0_ = inputs[kInputData]->GetDeviceShapeAdaptively();
  input_shape1_ = inputs[kInputX1]->GetDeviceShapeAdaptively();
  input_shape2_ = inputs[kInputX2]->GetDeviceShapeAdaptively();
  input_shape3_ = inputs[kInputValue]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kOutputData]->GetShapeVector();
  data_shape_size_ = SizeToLong(input_shape0_.size());
  inputx_shape_size_ = SizeToLong(input_shape1_.size());
  inputy_shape_size_ = SizeToLong(input_shape2_.size());
  value_shape_size_ = SizeToLong(input_shape3_.size());
  return KRET_OK;
}

bool AddcdivCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /* workspace */,
                                 const std::vector<AddressPtr> &outputs) {
  // check params
  if (dtype_ == kNumberTypeFloat32) {
    return AddcdivCheck<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    return AddcdivCheck<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    return AddcdivCheck<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    return AddcdivCheck<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'x' should be float16, float32, float64, int64, but got "
                      << TypeIdLabel(dtype_);
  }
}

template <typename T>
bool AddcdivCpuKernelMod::AddcdivCheck(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (dtype_value == kNumberTypeFloat16) {
    return AddcdivCompute<T, float16>(inputs, outputs);
  } else if (dtype_value == kNumberTypeFloat32) {
    return AddcdivCompute<T, float>(inputs, outputs);
  } else if (dtype_value == kNumberTypeFloat64) {
    return AddcdivCompute<T, double>(inputs, outputs);
  } else if (dtype_value == kNumberTypeInt32) {
    return AddcdivCompute<T, int>(inputs, outputs);
  } else if (dtype_value == kNumberTypeInt64) {
    return AddcdivCompute<T, int64_t>(inputs, outputs);
  }
  return true;
}

template <typename T>
T abs(T num) {
  if (num >= static_cast<T>(0.0)) {
    return num;
  } else {
    return -num;
  }
}

template <typename T1, typename T2>
bool AddcdivCpuKernelMod::AddcdivCompute(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto *input0 = static_cast<T1 *>(inputs[kInputData]->addr);
  const auto *input1 = static_cast<T1 *>(inputs[kInputX1]->addr);
  const auto *input2 = static_cast<T1 *>(inputs[kInputX2]->addr);
  const auto *input3 = static_cast<T2 *>(inputs[kInputValue]->addr);
  auto *output = static_cast<T1 *>(outputs[kOutputData]->addr);

  if ((inputx_shape_size_ + inputy_shape_size_ + value_shape_size_ + data_shape_size_) == 0) {
    auto eps_if_zero = static_cast<T1>(1e-6);
    auto zero = static_cast<T1>(0);
    if (abs(input2[0]) <= eps_if_zero) {
      auto prod = static_cast<T1>(input3[0]) * input1[0];
      if (abs(prod) <= eps_if_zero) {
        output[0] = std::numeric_limits<T1>::quiet_NaN();
        return true;
      }
      if (std::numeric_limits<T1>::has_infinity) {
        output[0] = prod > zero ? std::numeric_limits<T1>::infinity() : -std::numeric_limits<T1>::infinity();
      } else {
        output[0] = prod > zero ? std::numeric_limits<T1>::max() : std::numeric_limits<T1>::min();
      }
    } else {
      output[0] = static_cast<T1>(input3[0]) * input1[0] / input2[0] + input0[0];
    }
  } else {
    MultipleBroadcastIterator multi_broadcast_iterator({input_shape0_, input_shape1_, input_shape2_, input_shape3_},
                                                       output_shape_);
    auto base_task = [&input0, &input1, &input2, &input3, &output, &multi_broadcast_iterator](size_t start,
                                                                                              size_t end) {
      auto iter = multi_broadcast_iterator;
      iter.SetPos(start);
      auto eps_if_zero = static_cast<T1>(1e-6);
      auto zero = static_cast<T1>(0);
      for (size_t i = start; i < end; i++) {
        if (abs(input2[iter.GetInputPos(kIndex2)]) <= eps_if_zero) {
          auto prod = static_cast<T1>(input3[iter.GetInputPos(kIndex3)]) * input1[iter.GetInputPos(kIndex1)];
          if (abs(prod) <= eps_if_zero) {
            output[i] = std::numeric_limits<T1>::quiet_NaN();
            iter.GenNextPos();
            continue;
          }
          if (std::numeric_limits<T1>::has_infinity) {
            output[i] = prod > zero ? std::numeric_limits<T1>::infinity() : -std::numeric_limits<T1>::infinity();
          } else {
            output[i] = prod > zero ? std::numeric_limits<T1>::max() : std::numeric_limits<T1>::min();
          }
          iter.GenNextPos();
          continue;
        } else {
          output[i] = static_cast<T1>(input3[iter.GetInputPos(kIndex3)]) * input1[iter.GetInputPos(kIndex1)] /
                        input2[iter.GetInputPos(kIndex2)] +
                      input0[iter.GetInputPos(kIndex0)];
          iter.GenNextPos();
        }
      }
    };
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= static_cast<size_t>(output_shape_[i]);
    }
    ParallelLaunchAutoSearch(base_task, output_size_, this, &parallel_search_info_);
  }
  return true;
}

std::vector<KernelAttr> AddcdivCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Float32, Float32, Float32, Float16, Float32), ADD_KERNEL(Float32, Float32, Float32, Float32, Float32),
    ADD_KERNEL(Float32, Float32, Float32, Float64, Float32), ADD_KERNEL(Float32, Float32, Float32, Int32, Float32),
    ADD_KERNEL(Float32, Float32, Float32, Int64, Float32),   ADD_KERNEL(Float64, Float64, Float64, Float16, Float64),
    ADD_KERNEL(Float64, Float64, Float64, Float32, Float64), ADD_KERNEL(Float64, Float64, Float64, Float64, Float64),
    ADD_KERNEL(Float64, Float64, Float64, Int32, Float64),   ADD_KERNEL(Float64, Float64, Float64, Int64, Float64),
    ADD_KERNEL(Float16, Float16, Float16, Float16, Float16), ADD_KERNEL(Float16, Float16, Float16, Float32, Float16),
    ADD_KERNEL(Float16, Float16, Float16, Float64, Float16), ADD_KERNEL(Float16, Float16, Float16, Int32, Float16),
    ADD_KERNEL(Float16, Float16, Float16, Int64, Float16),   ADD_KERNEL(Int64, Int64, Int64, Float16, Int64),
    ADD_KERNEL(Int64, Int64, Int64, Float32, Int64),         ADD_KERNEL(Int64, Int64, Int64, Float64, Int64),
    ADD_KERNEL(Int64, Int64, Int64, Int32, Int64),           ADD_KERNEL(Int64, Int64, Int64, Int64, Int64)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Addcdiv, AddcdivCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
