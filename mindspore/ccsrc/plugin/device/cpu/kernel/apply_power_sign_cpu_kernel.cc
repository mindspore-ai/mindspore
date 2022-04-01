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

#include "plugin/device/cpu/kernel/apply_power_sign_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPowerSignInputsNum = 7;
constexpr size_t kPowerSignOutputsNum = 2;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexLr = 2;
constexpr size_t kIndexLogBase = 3;
constexpr size_t kIndexSignDecay = 4;
constexpr size_t kIndexBeta = 5;
constexpr size_t kIndexGrad = 6;

template <typename T>
int Sgn(T x) {
  if (x > T(0)) {
    return 1;
  }
  if (x < T(0)) {
    return -1;
  }
  return 0;
}
}  // namespace

template <typename T>
void ApplyPowerSignCpuKernelMod::LaunchPowerSign(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[kIndexVar]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndexM]->addr);
  T *lr = reinterpret_cast<T *>(inputs[kIndexLr]->addr);
  T *logbase = reinterpret_cast<T *>(inputs[kIndexLogBase]->addr);
  T *sign_decay = reinterpret_cast<T *>(inputs[kIndexSignDecay]->addr);
  T *beta = reinterpret_cast<T *>(inputs[kIndexBeta]->addr);
  T *gradient = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);

  // multithreading
  size_t lens = inputs[kIndexVar]->size > 0 ? static_cast<size_t>(inputs[kIndexVar]->size / sizeof(T)) : 1;
  auto task = [this, &var, &m, &gradient, &lr, &beta, &logbase, &sign_decay](size_t start, size_t end) {
    T one = static_cast<T>(1.0);
    for (size_t i = start; i < end; i++) {
      m[i] = gradient[i] * (one - beta[0]) + m[i] * beta[0];
      T sign_value = static_cast<T>(Sgn(gradient[i]) * Sgn(m[i]));
      T update = exp(logbase[0] * sign_decay[0] * sign_value) * gradient[i];
      var[i] = var[i] - lr[i] * update;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

void ApplyPowerSignCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kPowerSignInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kPowerSignOutputsNum, kernel_name_);
}

bool ApplyPowerSignCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPowerSignInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPowerSignOutputsNum, kernel_name_);

  if (dtype_ == kNumberTypeFloat32) {
    LaunchPowerSign<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchPowerSign<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' should be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

std::vector<KernelAttr> ApplyPowerSignCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddOutputAttr(kNumberTypeFloat16)
                                                       .AddOutputAttr(kNumberTypeFloat16)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyPowerSign, ApplyPowerSignCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
