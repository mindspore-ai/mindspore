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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/random_util.h"

namespace mindspore {
namespace kernel {
class MultinomialWithReplacementCpuKernelMod : public NativeCpuKernelMod {
 public:
  MultinomialWithReplacementCpuKernelMod() = default;
  ~MultinomialWithReplacementCpuKernelMod() override = default;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool CheckMultinomialWithReplacementShape();

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using MultinomialWithReplacementFunc =
    std::function<bool(MultinomialWithReplacementCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;

  template <typename T>
  int64_t *TrueCompute(T *in, int64_t *out, T *RandomData, int64_t i, int64_t num_col_);

  template <typename T>
  int64_t *FalseCompute(T *in, int64_t *out, T *RandomData, int64_t i, int64_t num_col_);

 private:
  random::MSPhiloxRandom generator_;
  using ResType = random::Array<uint32_t, random::MSPhiloxRandom::kResultElementCount>;
  ResType unused_results_;
  size_t used_result_index_ = random::MSPhiloxRandom::kResultElementCount;

  float RandFloat();
  uint64_t New64();
  void InitMSPhiloxRandom(int64_t seed, int64_t offset);
  uint32_t GenerateSingle();

  static std::vector<std::pair<KernelAttr, MultinomialWithReplacementFunc>> func_list_;
  MultinomialWithReplacementFunc kernel_func_;
  ShapeVector x_shape_;
  int64_t numsamples_;
  bool replacement_;
  int64_t num_row_;
  int64_t num_col_;
  BaseOperatorPtr kernel_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTINOMIAL_WITH_REPLACEMENT_CPU_KERNEL_H_
