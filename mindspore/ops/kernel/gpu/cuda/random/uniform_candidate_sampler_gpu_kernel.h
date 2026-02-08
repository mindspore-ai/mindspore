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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_CANDIDATE_SAMPLER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_CANDIDATE_SAMPLER_GPU_KERNEL_H_

#include <cmath>
#include <set>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/uniform_candidate_sampler_impl.cuh"

namespace mindspore {
namespace kernel {
class UniformCandidateSamplerGpuKernelMod : public NativeGpuKernelMod {
 public:
  UniformCandidateSamplerGpuKernelMod() = default;
  ~UniformCandidateSamplerGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  template <typename T>
  int64_t Sampling(const std::set<T> &set_input, std::vector<T> *sampled_candidates) {
    T tmp;
    T range;
    int64_t picked;
    int64_t counter = 0;
    std::set<T> set_container;
    // pick between [0, range_max_-1]
    if (std::numeric_limits<T>::max() < std::numeric_limits<int64_t>::max() &&
        range_max_ > static_cast<int64_t>(std::numeric_limits<T>::max())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', range_max_ failed to cast";
    }
    range = static_cast<T>(range_max_);
    std::uniform_int_distribution<T> distribution(0, range - 1);
    sampled_candidates->clear();
    if (unique_) {
      picked = 0;
      while (picked < num_sampled_) {
        tmp = distribution(rng_);
        counter++;
        if ((set_container.find(tmp) == set_container.end()) &&
            ((!remove_accidental_hits_) || set_input.find(tmp) == set_input.end())) {
          set_container.insert(tmp);
          sampled_candidates->push_back(tmp);
          picked++;
        }
      }
    } else {
      for (int64_t i = 0; i < num_sampled_; i++) {
        sampled_candidates->push_back(distribution(rng_));
      }
      counter = num_sampled_;
    }
    return counter;
  }

  template <typename S>
  S Probability() {
    S range;
    if (std::numeric_limits<S>::max() < std::numeric_limits<int64_t>::max() &&
        range_max_ > static_cast<int64_t>(std::numeric_limits<S>::max())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', range_max_ failed to cast";
    }
    range = static_cast<S>(range_max_);
    MS_EXCEPTION_IF_ZERO("range", range);
    return static_cast<S>(1.0f / range);
  }

  template <typename S>
  S ApproximateExpectedCount(S p, int64_t sampled_size, int64_t counter) {
    if (sampled_size == counter) return p * sampled_size;
    return -std::expm1(counter * std::log1p(-p));
  }

  using UCSGpuLaunchFunc = std::function<bool(
    UniformCandidateSamplerGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  UCSGpuLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UCSGpuLaunchFunc>> func_list_;
  int64_t num_true_{0};
  int64_t num_sampled_{0};
  bool unique_{false};
  int64_t range_max_{0};
  size_t input_size_{0};
  bool remove_accidental_hits_{false};
  bool is_null_input_{false};
  std::default_random_engine rng_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_CANDIDATE_SAMPLER_GPU_KERNEL_H_
