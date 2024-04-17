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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_SHUFFLE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_SHUFFLE_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <random>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class RandomShuffleCpuKernelMod : public NativeCpuKernelMod {
 public:
  RandomShuffleCpuKernelMod() = default;
  ~RandomShuffleCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T>
  bool ScalarShuffle(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs, const std::vector<size_t> &perm) const;

  template <typename T>
  bool ScalarShuffleWithBatchRank(const std::vector<kernel::KernelTensor *> &inputs,
                                  const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T>
  bool TensorShuffle(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs, const std::vector<size_t> &perm);

  template <typename T>
  bool TensorShuffleWithBatchRank(const std::vector<kernel::KernelTensor *> &inputs,
                                  const std::vector<kernel::KernelTensor *> &outputs);

  using RandomShuffleFunc = std::function<bool(RandomShuffleCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                               const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, RandomShuffleFunc>> func_list_;
  RandomShuffleFunc kernel_func_;

  int64_t outer_size_{1};
  int64_t inner_size_{1};
  size_t shuffle_size_{1};
  size_t batch_rank_{0};
  std::vector<int64_t> input_shape_;
  std::default_random_engine rng_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_SHUFFLE_CPU_KERNEL_H_
