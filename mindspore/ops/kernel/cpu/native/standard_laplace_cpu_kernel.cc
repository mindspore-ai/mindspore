/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/standard_laplace_cpu_kernel.h"
#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
namespace standard_laplace_cpu {
namespace {
constexpr size_t kStandardLaplaceInputsNum = 1;
constexpr size_t kStandardLaplaceOutputsNum = 1;
}  // namespace
bool StandardLaplaceCpuKernelMod::Init(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) {
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed2")));
  uint64_t init_seed = random::GetSeed(seed, seed2);
  rng_.seed(init_seed);
  return true;
}

bool StandardLaplaceCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                         const std::vector<kernel::KernelTensor *> &,
                                         const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kStandardLaplaceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStandardLaplaceOutputsNum, kernel_name_);

  // Init output address.
  auto output = GetDeviceAddress<float>(outputs, kIndex0);

  // Init sample number.
  size_t num_sample = outputs[kIndex0]->size() / sizeof(float);

  // Uniform variates sampled from the open-interval (-1,1) rather than [-1, 1].
  float lo = std::nextafter(-1.f, 0.f);
  std::uniform_real_distribution<float> distribution(lo, 1.0);

  // Generate random laplace values.
  for (size_t i = 0; i < num_sample; ++i) {
    float uniform_random_num = distribution(rng_);
    float uniform_random_num_sign = std::copysignf(1.0, uniform_random_num);
    output[i] = static_cast<float>(-uniform_random_num_sign * std::log(1.0 - std::abs(uniform_random_num)));
  }

  return true;
}

std::vector<KernelAttr> StandardLaplaceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, StandardLaplace, StandardLaplaceCpuKernelMod);
}  // namespace standard_laplace_cpu
}  // namespace kernel
}  // namespace mindspore
