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

#include "plugin/device/cpu/kernel/cauchy_cpu_kernel.h"
#include <vector>
#include <memory>
#include <random>
#include "mindspore/core/ops/cauchy.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCauchyOutputNum = 1;
}  // namespace

bool CauchyCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCauchyOutputNum, kernel_name_);
  sigma_ = GetValue<float>(primitive_->GetAttr("sigma"));
  median_ = GetValue<float>(primitive_->GetAttr("median"));
  return true;
}

bool CauchyCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &,
                                const std::vector<kernel::KernelTensor *> &,
                                const std::vector<kernel::KernelTensor *> &outputs) {
  (void)LaunchKernel<float>(outputs);
  return true;
}

template <typename T>
bool CauchyCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &outputs) const {
  T *y_data = static_cast<T *>(outputs[0]->device_ptr());
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::cauchy_distribution<float> cauchy_d(median_, sigma_);
  auto end = outputs[0]->size() / sizeof(T);

  for (size_t i = 0; i < end; ++i) {
    float data = cauchy_d(generator);
    y_data[i] = static_cast<T>(data);
  }

  return true;
}

std::vector<KernelAttr> CauchyCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr().AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Cauchy, CauchyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
