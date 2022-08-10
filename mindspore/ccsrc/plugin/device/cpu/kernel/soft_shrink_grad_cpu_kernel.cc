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

#include "plugin/device/cpu/kernel/soft_shrink_grad_cpu_kernel.h"
#include "mindspore/core/ops/grad/soft_shrink_grad.h"
#include "plugin/device/cpu/kernel/nnacl/fp32_grad/activation_grad_fp32.h"

namespace mindspore {
namespace kernel {
#define SOFT_SHRINK_GRAD_CPU_REGISTER(DT, T) \
  KernelAttr().AddInputAttr(DT).AddInputAttr(DT).AddOutputAttr(DT), &SoftShrinkGradCpuKernelMod::LaunchKernel<T>

template <typename T>
bool SoftShrinkGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  /* float optimize */
  if (std::is_same_v<T, float>) {
    float *src0 = reinterpret_cast<float *>(inputs.at(kIndex0)->addr);
    float *src1 = reinterpret_cast<float *>(inputs.at(kIndex1)->addr);
    float *out = reinterpret_cast<float *>(outputs.at(kIndex0)->addr);

    auto task = [src0, src1, out, this](size_t start, size_t end) {
      auto src0_tmp = src0 + start;
      auto src1_tmp = src1 + start;
      auto out_tmp = out + start;
      (void)SoftShrinkGrad(src0_tmp, src1_tmp, (end - start), out_tmp, this->lambd_);
    };
    ParallelLaunchAutoSearch(task, size_, this, &parallel_search_info_);
    return true;
  }

  /* common soft shrink grad */
  T *dy_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  T *x_addr = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  T *dx_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  T lambd_value = static_cast<T>(lambd_);
  auto task = [dy_addr, x_addr, dx_addr, lambd_value](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      dx_addr[i] = (x_addr[i] >= -lambd_value && x_addr[i] <= lambd_value) ? 0 : dy_addr[i];
    }
  };
  ParallelLaunchAutoSearch(task, size_, this, &parallel_search_info_);

  return true;
}

bool SoftShrinkGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::SoftShrinkGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast SoftShrinkGrad ops failed!";
    return false;
  }
  lambd_ = kernel_ptr->get_lambd();

  if (auto ret = MatchKernelFunc(base_operator, inputs, outputs); !ret) {
    return ret;
  }
  return true;
}

int SoftShrinkGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  auto in_shape = inputs[kIndex0]->GetShapeVector();
  size_ = std::accumulate(in_shape.begin(), in_shape.end(), size_t(1), std::multiplies<size_t>());
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, SoftShrinkGradCpuKernelMod::KernelRunFunc>>
  &SoftShrinkGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SoftShrinkGradCpuKernelMod::KernelRunFunc>> func_list = {
    {SOFT_SHRINK_GRAD_CPU_REGISTER(kNumberTypeFloat32, float)},
    {SOFT_SHRINK_GRAD_CPU_REGISTER(kNumberTypeInt32, int32_t)},
    {SOFT_SHRINK_GRAD_CPU_REGISTER(kNumberTypeInt64, int64_t)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SoftShrinkGrad, SoftShrinkGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
