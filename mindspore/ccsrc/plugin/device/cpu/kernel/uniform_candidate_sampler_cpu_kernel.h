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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UNIFORM_CANDIDATE_SAMPLER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UNIFORM_CANDIDATE_SAMPLER_CPU_KERNEL_H_

#include <cmath>
#include <unordered_set>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include <random>
#include <limits>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class UniformCandidateSamplerCpuKernelMod : public NativeCpuKernelMod,
                                            public MatchKernelHelper<UniformCandidateSamplerCpuKernelMod> {
 public:
  UniformCandidateSamplerCpuKernelMod()
      : num_true_(0),
        num_sampled_(0),
        unique_(false),
        range_max_(0),
        input_size_(0),
        remove_accidental_hits_(false),
        is_null_input_(false) {}

  ~UniformCandidateSamplerCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspaces, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  void CheckAttribute();
  void CheckInputsAndOutputs(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);

  template <typename T>
  int64_t Sampling(T *sampled_candidates, const size_t length);

  template <typename S>
  void ExpectedLanuch(const int64_t counter, S *true_expected_count, S *sampled_expected_count);

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                    const std::vector<AddressPtr> &outputs);

  int64_t batch_rank_{0};
  int64_t batch_size_{1};
  int64_t num_true_;
  int64_t num_sampled_;
  bool unique_;
  int64_t range_max_;
  size_t input_size_;
  std::vector<size_t> output_sizes_;
  std::vector<size_t> output_steps_;

  bool remove_accidental_hits_;
  bool is_null_input_;

  std::default_random_engine rng_;
  std::unordered_set<int64_t> set_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UNIFORM_CANDIDATE_SAMPLER_CPU_KERNEL_H_
