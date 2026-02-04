/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility>

#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
namespace multinomial_with_replacement_cpu {
class MultinomialWithReplacementCpuKernelMod : public NativeCpuKernelMod {
 public:
  MultinomialWithReplacementCpuKernelMod() = default;
  ~MultinomialWithReplacementCpuKernelMod() override = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool CheckMultinomialWithReplacementShape();

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using MultinomialWithReplacementFunc =
    std::function<bool(MultinomialWithReplacementCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;

  template <typename T>
  int64_t *TrueCompute(T *in, int64_t *out, T *RandomData, int64_t i, int64_t num_col_) const;

  template <typename T>
  int64_t *FalseCompute(T *in, int64_t *out, T *RandomData, int64_t i, int64_t num_col_) const;

 private:
  random::PhiloxRandom generator_;
  using ResType = random::Array<uint32_t, random::PhiloxRandom::kResultElementCount>;
  ResType unused_results_;
  size_t used_result_index_ = random::PhiloxRandom::kResultElementCount;

  float RandFloat();
  uint64_t New64() const;
  void InitPhiloxRandom(int64_t seed, int64_t offset);
  uint32_t GenerateSingle();

  static std::vector<std::pair<KernelAttr, MultinomialWithReplacementFunc>> func_list_;
  MultinomialWithReplacementFunc kernel_func_;
  ShapeVector x_shape_;
  int64_t numsamples_;
  bool replacement_;
  bool init_state_{true};
  int64_t init_seed_{0};
  int64_t init_offset_{0};
  BaseOperatorPtr kernel_ptr_{nullptr};
};
}  // namespace multinomial_with_replacement_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL_H_
