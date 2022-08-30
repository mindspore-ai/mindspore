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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESERVOIR_REPLAY_BUFFER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESERVOIR_REPLAY_BUFFER_CPU_KERNEL_H_
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/rl/reservoir_replay_buffer.h"

namespace mindspore {
namespace kernel {
class ReservoirReplayBufferCreateCpuKernel : public NativeCpuKernelMod {
 public:
  ReservoirReplayBufferCreateCpuKernel() = default;
  ~ReservoirReplayBufferCreateCpuKernel() override = default;

  // Collect and prepare kernel algorithm parameter.
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
  std::shared_ptr<ReservoirReplayBuffer> reservoir_replay_buffer_{nullptr};
};

class ReservoirReplayBufferPushCpuKernel : public NativeCpuKernelMod {
 public:
  ReservoirReplayBufferPushCpuKernel() = default;
  ~ReservoirReplayBufferPushCpuKernel() override = default;

  // Init kernel from CNode.
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() {
    static const std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
  std::shared_ptr<ReservoirReplayBuffer> reservoir_replay_buffer_{nullptr};
};

class ReservoirReplayBufferSampleCpuKernel : public NativeCpuKernelMod {
 public:
  ReservoirReplayBufferSampleCpuKernel() = default;
  ~ReservoirReplayBufferSampleCpuKernel() override = default;

  // Init kernel from CNode.
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() {
    static const std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
  size_t batch_size_{0};
  std::vector<size_t> schema_;
  std::shared_ptr<ReservoirReplayBuffer> reservoir_replay_buffer_{nullptr};
};

class ReservoirReplayBufferDestroyCpuKernel : public NativeCpuKernelMod {
 public:
  ReservoirReplayBufferDestroyCpuKernel() = default;
  ~ReservoirReplayBufferDestroyCpuKernel() override = default;

  // Collect and prepare kernel algorithm parameter.
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESERVOIR_REPLAY_BUFFER_CPU_KERNEL_H_
