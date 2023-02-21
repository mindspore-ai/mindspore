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

#include "plugin/device/cpu/kernel/apply_momentum_cpu_kernel.h"
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyMomentumInputsNum = 5;
}  // namespace

bool ApplyMomentumCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = base_operator->name();
  dtype_ = inputs[0]->GetDtype();
  return true;
}

int ApplyMomentumCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  return KernelMod::Resize(base_operator, inputs, outputs);
}

bool ApplyMomentumCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyMomentumInputsNum, kernel_name_);
  if (inputs[0]->size != inputs[1]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of input 'accumulation' and 'variable' must be "
                         "same, but got the memory size of 'accumulation': "
                      << inputs[1]->size << " and 'variable': " << inputs[0]->size;
  }
  if (inputs[0]->size != inputs[3]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of input 'gradient' and 'variable' must be "
                         "same, but got the memory size of 'gradient': "
                      << inputs[3]->size << " and 'variable': " << inputs[0]->size;
  }

  if (dtype_ == kNumberTypeFloat32) {
    LaunchApplyMomentum<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchApplyMomentum<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchApplyMomentum<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    LaunchApplyMomentum<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    LaunchApplyMomentum<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    LaunchApplyMomentum<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    LaunchApplyMomentum<uint16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    LaunchApplyMomentum<uint32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchApplyMomentum<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchApplyMomentum<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    LaunchApplyMomentum<uint64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchApplyMomentum<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchApplyMomentum<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dtype of 'var' should be float64, int64, float, float16, int16, int32, int8, uint16, "
                     "uint32, uint64, uint8, complex64, complex128, but get "
                  << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename T>
void ApplyMomentumCpuKernelMod::LaunchApplyMomentum(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &) {
  T *weight = reinterpret_cast<T *>(inputs[0]->addr);
  T *accumulate = reinterpret_cast<T *>(inputs[1]->addr);
  T learning_rate = reinterpret_cast<T *>(inputs[2]->addr)[0];
  const T *gradient = reinterpret_cast<T *>(inputs[3]->addr);
  T moment = reinterpret_cast<T *>(inputs[4]->addr)[0];
  size_t elem_num = inputs[0]->size / sizeof(T);

  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      accumulate[i] = accumulate[i] * moment + gradient[i];
      weight[i] -= accumulate[i] * learning_rate;
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
}

#define ADD_KERNEL_1(dtype)              \
  {                                      \
    KernelAttr()                         \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddOutputAttr(kNumberType##dtype) \
      .AddOutInRef(0, 0)                 \
  }
#define ADD_KERNEL_2(dtype)              \
  {                                      \
    KernelAttr()                         \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddInputAttr(kNumberType##dtype)  \
      .AddOutputAttr(kNumberType##dtype) \
      .AddOutputAttr(kNumberType##dtype) \
      .AddOutInRef(0, 0)                 \
  }

std::vector<KernelAttr> ApplyMomentumCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL_1(Float32),    ADD_KERNEL_2(Float32),   ADD_KERNEL_1(Float16),   ADD_KERNEL_2(Float16),
    ADD_KERNEL_1(Int8),       ADD_KERNEL_2(Int8),      ADD_KERNEL_1(UInt8),     ADD_KERNEL_2(UInt8),
    ADD_KERNEL_1(Int16),      ADD_KERNEL_2(Int16),     ADD_KERNEL_1(UInt16),    ADD_KERNEL_2(UInt16),
    ADD_KERNEL_1(UInt32),     ADD_KERNEL_2(UInt32),    ADD_KERNEL_1(Int32),     ADD_KERNEL_2(Int32),
    ADD_KERNEL_1(Int64),      ADD_KERNEL_2(Int64),     ADD_KERNEL_1(UInt64),    ADD_KERNEL_2(UInt64),
    ADD_KERNEL_1(Float64),    ADD_KERNEL_2(Float64),   ADD_KERNEL_1(Complex64), ADD_KERNEL_2(Complex64),
    ADD_KERNEL_1(Complex128), ADD_KERNEL_2(Complex128)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyMomentum, ApplyMomentumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
