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

#include "plugin/device/gpu/kernel/sponge/angle/angle_atom_energy_kernel.h"
#include "ops/angle_atom_energy.h"

namespace mindspore {
namespace kernel {
using KernelRunFunc = AngleAtomEnergyGpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &AngleAtomEnergyGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &AngleAtomEnergyGpuKernelMod::LaunchKernel<float, int>}};
  return func_list;
}

bool AngleAtomEnergyGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::AngleAtomEnergy>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  kernel_name_ = kernel_ptr->name();
  angle_numbers_ = static_cast<int>(kernel_ptr->get_angle_numbers());
  return true;
}

int AngleAtomEnergyGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto shape_uint_crd = inputs[0]->GetShapeVector();
  ele_uint_crd_ = SizeOf(shape_uint_crd);
  return KRET_OK;
}

template <typename T, typename T1>
bool AngleAtomEnergyGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs) {
  auto uint_crd_f = GetDeviceAddress<const T1>(inputs, 0);
  auto scaler_f = GetDeviceAddress<T>(inputs, 1);
  auto atom_a = GetDeviceAddress<const T1>(inputs, 2);
  auto atom_b = GetDeviceAddress<const T1>(inputs, 3);
  auto atom_c = GetDeviceAddress<const T1>(inputs, 4);
  auto angle_k = GetDeviceAddress<T>(inputs, 5);
  auto angle_theta0 = GetDeviceAddress<T>(inputs, 6);

  auto ene = GetDeviceAddress<T>(outputs, 0);
  AngleAtomEnergy(angle_numbers_, ele_uint_crd_, uint_crd_f, scaler_f, atom_a, atom_b, atom_c, angle_k, angle_theta0,
                  ene, reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}

bool AngleAtomEnergyGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AngleAtomEnergy, AngleAtomEnergyGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
