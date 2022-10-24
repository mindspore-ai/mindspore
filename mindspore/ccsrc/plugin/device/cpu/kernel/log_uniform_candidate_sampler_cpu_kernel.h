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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CANDIDATE_SAMPLER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CANDIDATE_SAMPLER_CPU_KERNEL_H_

#include <functional>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/random_util.h"

namespace mindspore {
namespace kernel {
class LogUniformCandidateSamplerCpuKernel : public NativeCpuKernelMod {
 public:
  LogUniformCandidateSamplerCpuKernel() = default;
  ~LogUniformCandidateSamplerCpuKernel() = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override {
    std::vector<KernelAttr> support = {KernelAttr()
                                         .AddInputAttr(kNumberTypeInt64)
                                         .AddOutputAttr(kNumberTypeInt64)
                                         .AddOutputAttr(kNumberTypeFloat32)
                                         .AddOutputAttr(kNumberTypeFloat32)};
    return support;
  }

 private:
  int64_t Sample(random::SinglePhiloxRandom *single);
  float Probability(int64_t value) const;

 private:
  int32_t num_true_;
  int32_t num_sampled_;
  bool unique_;
  int32_t range_max_;
  random::GuardedPhiloxRandom generator_;
  double log_range_;
  int32_t seed_;
  int64_t reserveSamplesNr_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CANDIDATE_SAMPLER_CPU_KERNEL_H_
