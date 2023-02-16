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
#include "plugin/device/cpu/kernel/nnacl/arithmetic.h"
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

template <typename T1, typename T2>
bool AddcdivCpuKernelMod::AddcdivCompute(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto *input0 = static_cast<T1 *>(inputs[kInputData]->addr);
  const auto *input1 = static_cast<T1 *>(inputs[kInputX1]->addr);
  const auto *input2 = static_cast<T1 *>(inputs[kInputX2]->addr);
  const auto *input3 = static_cast<T2 *>(inputs[kInputValue]->addr);
  auto *output = static_cast<T1 *>(outputs[kOutputData]->addr);
  AddcdivMul(input1, input3, output);
  AddcdivDiv(output, input2, output);
  AddcdivAdd(input0, output, output);
  return true;
}

template <typename T1, typename T2>
void AddcdivCpuKernelMod::AddcdivMul(const T1 *input1, const T2 *input2, T1 *output) {
  if ((inputx_shape_size_ + inputy_shape_size_ + value_shape_size_) == 0) {
    T1 mul2 = static_cast<T1>(input2[0]);
    output[0] = input1[0] * mul2;
  } else {
    BroadcastIterator mul_iter(input_shape1_, input_shape3_, output_shape_);
    auto mul_task = [&input1, &input2, &output, &mul_iter](size_t mul_start, size_t mul_end) {
      auto iter = mul_iter;
      iter.SetPos(mul_start);
      for (auto i = mul_start; i < mul_end; i++) {
        T1 mul2 = static_cast<T1>(input2[iter.GetInputPosB()]);
        output[i] = input1[iter.GetInputPosA()] * mul2;
        iter.GenNextPos();
      }
    };
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= static_cast<int64_t>(output_shape_[i]);
    }
    ParallelLaunchAutoSearch(mul_task, LongToSize(output_size_), this, &parallel_search_info_);
  }
}

template <typename T>
void AddcdivCpuKernelMod::AddcdivAdd(const T *input1, const T *input2, T *output) {
  if ((inputx_shape_size_ + inputy_shape_size_ + value_shape_size_ + data_shape_size_) == 0) {
    output[0] = input1[0] + input2[0];
  } else {
    BroadcastIterator add_iter(input_shape0_, output_shape_, output_shape_);
    auto add_task = [&input1, &input2, &output, &add_iter](size_t add_start, size_t add_end) {
      auto iter = add_iter;
      iter.SetPos(add_start);
      for (size_t i = add_start; i < add_end; i++) {
        output[i] = input1[iter.GetInputPosA()] + input2[iter.GetInputPosB()];
        iter.GenNextPos();
      }
    };
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= static_cast<int64_t>(output_shape_[i]);
    }
    ParallelLaunchAutoSearch(add_task, LongToSize(output_size_), this, &parallel_search_info_);
  }
}

template <typename T>
T abs(T num) {
  if (num >= static_cast<T>(0.0)) {
    return num;
  } else {
    return -num;
  }
}

template <typename T>
void AddcdivCpuKernelMod::AddcdivDiv(const T *input1, const T *input2, T *output) {
  if (inputx_shape_size_ == 0 && inputy_shape_size_ == 0) {
    const auto eps_if_zero = static_cast<T>(1e-6);
    auto zero = static_cast<T>(0);
    if (abs(input2[0] - zero) <= eps_if_zero) {
      if (abs(input1[0] - zero) <= eps_if_zero) {
        output[0] = std::numeric_limits<T>::quiet_NaN();
        return;
      }
      if (std::numeric_limits<T>::has_infinity) {
        output[0] = input1[0] > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        output[0] = input1[0] > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
    }
  } else {
    BroadcastIterator div_iter(output_shape_, input_shape2_, output_shape_);
    auto div_task = [&input1, &input2, &output, &div_iter](int64_t div_start, int64_t div_end) {
      const auto eps_if_zero = static_cast<T>(1e-6);
      auto iter = div_iter;
      iter.SetPos(div_start);
      for (int64_t i = div_start; i < div_end; i++) {
        auto zero = static_cast<T>(0);
        auto addcdiv_dividend = input1[iter.GetInputPosA()];
        auto addcdiv_divisor = input2[iter.GetInputPosB()];
        if (abs(addcdiv_divisor - zero) <= eps_if_zero) {
          if (abs(addcdiv_dividend - zero) <= eps_if_zero) {
            output[i] = std::numeric_limits<T>::quiet_NaN();
            iter.GenNextPos();
            continue;
          }
          if (std::numeric_limits<T>::has_infinity) {
            output[i] =
              addcdiv_dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
          } else {
            output[i] = addcdiv_dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
          }
          iter.GenNextPos();
          continue;
        }
        output[i] = addcdiv_dividend / addcdiv_divisor;
        iter.GenNextPos();
      }
    };
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= static_cast<int64_t>(output_shape_[i]);
    }
    ParallelLaunchAutoSearch(div_task, LongToSize(output_size_), this, &parallel_search_info_);
  }
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
