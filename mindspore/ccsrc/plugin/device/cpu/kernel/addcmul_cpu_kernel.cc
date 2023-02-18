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

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/addcmul_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/mul_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
#define F32 kNumberTypeFloat32
#define F16 kNumberTypeFloat16
#define U8 kNumberTypeUInt8
#define I8 kNumberTypeInt8
#define I32 kNumberTypeInt32
#define I64 kNumberTypeInt64
#define F64 kNumberTypeFloat64
const size_t kOutputNum = 1;
const size_t kInputNum = 4;
const size_t kInputData = 0;
const size_t kInputX1 = 1;
const size_t kInputX2 = 2;
const size_t kInputValue = 3;
const size_t kOutputData = 0;
}  // namespace

bool AddcmulCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  dtype_ = inputs[kIndex0]->GetDtype();
  return true;
}

int AddcmulCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs[kInputData]->GetDtype();
  dtype_value_ = inputs[kInputValue]->GetDtype();
  input_shape0_ = inputs[kInputData]->GetDeviceShapeAdaptively();
  input_shape1_ = inputs[kInputX1]->GetDeviceShapeAdaptively();
  input_shape2_ = inputs[kInputX2]->GetDeviceShapeAdaptively();
  input_shape3_ = inputs[kInputValue]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kOutputData]->GetShapeVector();
  data_shape_size_ = input_shape0_.size();
  inputx_shape_size_ = input_shape1_.size();
  inputy_shape_size_ = input_shape2_.size();
  value_shape_size_ = input_shape3_.size();
  return KRET_OK;
}

template <typename T>
void AddcmulCpuKernelMod::AddcmulMul1(const T *input1, const T *input2, T *output) {
  if ((inputx_shape_size_ + inputy_shape_size_ + value_shape_size_) == 0) {
    output[0] = static_cast<T>(input1[0] * input2[0]);
  } else {
    BroadcastIterator mul_iter(input_shape1_, input_shape2_, output_shape_);
    auto mul_task = [&input1, &input2, &output, &mul_iter](size_t mul_start, size_t mul_end) {
      auto iter = mul_iter;
      iter.SetPos(mul_start);
      for (size_t i = mul_start; i < mul_end; i++) {
        output[i] = static_cast<T>(input1[iter.GetInputPosA()] * input2[iter.GetInputPosB()]);
        iter.GenNextPos();
      }
    };
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= static_cast<size_t>(output_shape_[i]);
    }
    ParallelLaunchAutoSearch(mul_task, output_size_, this, &parallel_search_info_);
  }
}

template <typename T1, typename T2>
void AddcmulCpuKernelMod::AddcmulMul2(const T2 *input1, const T1 *input2, T1 *output) {
  if ((inputx_shape_size_ + inputy_shape_size_ + value_shape_size_) == 0) {
    output[0] = static_cast<T1>(input1[0]) * input2[0];
  } else {
    BroadcastIterator base_iter(input_shape3_, output_shape_, output_shape_);
    auto task = [&input1, &input2, &output, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        output[i] = static_cast<T1>(input1[iter.GetInputPosA()]) * input2[iter.GetInputPosB()];
        iter.GenNextPos();
      }
    };
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= static_cast<size_t>(output_shape_[i]);
    }
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }
}

template <typename T>
void AddcmulCpuKernelMod::AddcmulAdd(const T *input1, const T *input2, T *output) {
  if ((inputx_shape_size_ + inputy_shape_size_ + value_shape_size_ + data_shape_size_) == 0) {
    output[0] = static_cast<T>(input1[0] + input2[0]);
  } else {
    BroadcastIterator base_iter(input_shape0_, output_shape_, output_shape_);
    auto add_task = [&input1, &input2, &output, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        output[i] = static_cast<T>(input1[iter.GetInputPosA()] + input2[iter.GetInputPosB()]);
        iter.GenNextPos();
      }
    };
    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= static_cast<size_t>(output_shape_[i]);
    }
    ParallelLaunchAutoSearch(add_task, output_size_, this, &parallel_search_info_);
  }
}

template <typename T>
bool AddcmulCpuKernelMod::AddcmulCheck(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (dtype_value_ == kNumberTypeFloat16) {
    return AddcmulCompute<T, float16>(inputs, outputs);
  } else if (dtype_value_ == kNumberTypeFloat32) {
    return AddcmulCompute<T, float>(inputs, outputs);
  } else if (dtype_value_ == kNumberTypeFloat64) {
    return AddcmulCompute<T, double>(inputs, outputs);
  } else if (dtype_value_ == kNumberTypeInt32) {
    return AddcmulCompute<T, int>(inputs, outputs);
  } else if (dtype_value_ == kNumberTypeInt64) {
    return AddcmulCompute<T, int64_t>(inputs, outputs);
  }
  return true;
}

template <typename T1, typename T2>
bool AddcmulCpuKernelMod::AddcmulCompute(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto *input0 = static_cast<T1 *>(inputs[kInputData]->addr);
  const auto *input1 = static_cast<T1 *>(inputs[kInputX1]->addr);
  const auto *input2 = static_cast<T1 *>(inputs[kInputX2]->addr);
  const auto *input3 = static_cast<T2 *>(inputs[kInputValue]->addr);
  auto *output = static_cast<T1 *>(outputs[kOutputData]->addr);

  AddcmulMul1(input1, input2, output);
  AddcmulMul2(input3, output, output);
  AddcmulAdd(input0, output, output);
  return true;
}

bool AddcmulCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /* workspace */,
                                 const std::vector<AddressPtr> &outputs) {
  // check params
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    return AddcmulCheck<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    return AddcmulCheck<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    return AddcmulCheck<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    return AddcmulCheck<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    return AddcmulCheck<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    return AddcmulCheck<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    return AddcmulCheck<int8_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'x' should be float16, float32, float64, int8, uint8,int32, int64, "
                         "but got "
                      << TypeIdLabel(dtype_);
  }
}

std::vector<KernelAttr> AddcmulCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F16).AddOutputAttr(F32),
    KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddOutputAttr(F32),
    KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I8).AddOutputAttr(F32),
    KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I32).AddOutputAttr(F32),
    KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(U8).AddOutputAttr(F32),
    KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F64).AddOutputAttr(F32),
    KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I64).AddOutputAttr(F32),
    KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F16).AddOutputAttr(F64),
    KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F32).AddOutputAttr(F64),
    KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I8).AddOutputAttr(F64),
    KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I32).AddOutputAttr(F64),
    KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(U8).AddOutputAttr(F64),
    KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddOutputAttr(F64),
    KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I64).AddOutputAttr(F64),
    KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddOutputAttr(F16),
    KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F32).AddOutputAttr(F16),
    KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I8).AddOutputAttr(F16),
    KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I32).AddOutputAttr(F16),
    KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(U8).AddOutputAttr(F16),
    KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F64).AddOutputAttr(F16),
    KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I64).AddOutputAttr(F16),
    KernelAttr().AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(F16).AddOutputAttr(I32),
    KernelAttr().AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(F32).AddOutputAttr(I32),
    KernelAttr().AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I8).AddOutputAttr(I32),
    KernelAttr().AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddOutputAttr(I32),
    KernelAttr().AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(U8).AddOutputAttr(I32),
    KernelAttr().AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(F64).AddOutputAttr(I32),
    KernelAttr().AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I32).AddInputAttr(I64).AddOutputAttr(I32),
    KernelAttr().AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(F16).AddOutputAttr(U8),
    KernelAttr().AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(F32).AddOutputAttr(U8),
    KernelAttr().AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(I8).AddOutputAttr(U8),
    KernelAttr().AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(I32).AddOutputAttr(U8),
    KernelAttr().AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddOutputAttr(U8),
    KernelAttr().AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(F64).AddOutputAttr(U8),
    KernelAttr().AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(U8).AddInputAttr(I64).AddOutputAttr(U8),
    KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(F16).AddOutputAttr(I64),
    KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(F32).AddOutputAttr(I64),
    KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I8).AddOutputAttr(I64),
    KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I32).AddOutputAttr(I64),
    KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(U8).AddOutputAttr(I64),
    KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(F64).AddOutputAttr(I64),
    KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddOutputAttr(I64),
    KernelAttr().AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(F16).AddOutputAttr(I8),
    KernelAttr().AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(F32).AddOutputAttr(I8),
    KernelAttr().AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddOutputAttr(I8),
    KernelAttr().AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I32).AddOutputAttr(I8),
    KernelAttr().AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(U8).AddOutputAttr(I8),
    KernelAttr().AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(F64).AddOutputAttr(I8),
    KernelAttr().AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(I8).AddInputAttr(F32).AddOutputAttr(I8)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Addcmul, AddcmulCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
