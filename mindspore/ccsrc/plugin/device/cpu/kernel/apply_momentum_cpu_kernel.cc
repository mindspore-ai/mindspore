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
  return true;
}

int ApplyMomentumCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return static_cast<int>(KRET_OK);
}

bool ApplyMomentumCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &) {
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
  auto *weight = reinterpret_cast<float *>(inputs[0]->addr);
  auto *accumulate = reinterpret_cast<float *>(inputs[1]->addr);
  float learning_rate = reinterpret_cast<float *>(inputs[2]->addr)[0];
  const auto *gradient = reinterpret_cast<float *>(inputs[3]->addr);
  float moment = reinterpret_cast<float *>(inputs[4]->addr)[0];
  size_t elem_num = inputs[0]->size / sizeof(float);
  for (size_t i = 0; i < elem_num; ++i) {
    accumulate[i] = accumulate[i] * moment + gradient[i];
    weight[i] -= accumulate[i] * learning_rate;
  }
  return true;
}

std::vector<KernelAttr> ApplyMomentumCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyMomentum, ApplyMomentumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
