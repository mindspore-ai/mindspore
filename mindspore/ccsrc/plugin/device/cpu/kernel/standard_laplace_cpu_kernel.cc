/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <random>
#include <thread>
#include <memory>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/standard_laplace_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kStandardLaplaceInputsNum = 1;
constexpr size_t kStandardLaplaceOutputsNum = 1;
constexpr float kRandomBlockSize = 128.0;
}  // namespace
void StandardLaplace(float *output, std::uniform_real_distribution<float> distribution,
                     std::default_random_engine random_generator, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    float uniform_random_num = distribution(random_generator);
    float uniform_random_num_sign = std::copysignf(1.0, uniform_random_num);
    output[i] = static_cast<float>(-uniform_random_num_sign * std::log(1.0 - std::abs(uniform_random_num)));
  }
}

void LaunchStandardLaplace(StandardLaplaceCpuKernelMod *content, unsigned int seed,
                           const std::vector<AddressPtr> &outputs) {
  MS_ERROR_IF_NULL_WO_RET_VAL(content);
  MS_ERROR_IF_NULL_WO_RET_VAL(outputs[kIndex0]);
  MS_ERROR_IF_NULL_WO_RET_VAL(outputs[kIndex0]->addr);

  auto output = reinterpret_cast<float *>(outputs[kIndex0]->addr);
  // multithreading
  size_t lens = outputs[kIndex0]->size / sizeof(float);
  auto thread_pool = GetActorMgrInnerThreadPool();
  size_t max_thread_num = thread_pool->GetKernelThreadNum();
  size_t thread_num = lens < static_cast<size_t>(kRandomBlockSize) * max_thread_num
                        ? static_cast<size_t>(std::ceil(lens / kRandomBlockSize))
                        : max_thread_num;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;

  // Uniform variates sampled from the open-interval (-1,1) rather than [-1, 1].
  float lo = std::nextafter(-1.f, 0.f);
  std::uniform_real_distribution<float> distribution(lo, 1.0);

  auto task = [once_compute_size, seed, output, &distribution](size_t start, size_t end) {
    auto task_id = start / once_compute_size;
    std::default_random_engine random_generator(seed + task_id);
    StandardLaplace(output, distribution, random_generator, start, end);
  };
  ParallelLaunch(task, lens, kRandomBlockSize, content);
}

bool StandardLaplaceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &,
                                       const std::vector<KernelTensorPtr> &) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  seed_ = LongToInt(GetValue<int64_t>(prim->GetAttr("seed")));
  seed2_ = LongToInt(GetValue<int64_t>(prim->GetAttr("seed2")));
  return true;
}

bool StandardLaplaceCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kStandardLaplaceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStandardLaplaceOutputsNum, kernel_name_);
  unsigned int RNG_seed = 0;
  std::random_device rd;
  if (seed2_ != 0) {
    RNG_seed = IntToUint(seed2_);
  } else if (seed_ != 0) {
    RNG_seed = IntToUint(seed_);
  } else {
    RNG_seed = rd();
  }
  LaunchStandardLaplace(this, RNG_seed, outputs);
  return true;
}

std::vector<KernelAttr> StandardLaplaceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, StandardLaplace, StandardLaplaceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
