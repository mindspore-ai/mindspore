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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRIORITY_REPLAY_BUFFER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRIORITY_REPLAY_BUFFER_CPU_KERNEL_H_
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/rl/priority_replay_buffer.h"

namespace mindspore {
namespace kernel {
class PriorityReplayBufferCreateCpuKernel : public DeprecatedNativeCpuKernelMod {
 public:
  PriorityReplayBufferCreateCpuKernel() = default;
  ~PriorityReplayBufferCreateCpuKernel() override = default;

  // Collect and prepare kernel algorithm parameter.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferPushCpuKernel : public DeprecatedNativeCpuKernelMod {
 public:
  PriorityReplayBufferPushCpuKernel() = default;
  ~PriorityReplayBufferPushCpuKernel() override = default;

  // Init kernel from CNode.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  int64_t handle_{-1};
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferSampleCpuKernel : public DeprecatedNativeCpuKernelMod {
 public:
  PriorityReplayBufferSampleCpuKernel() = default;
  ~PriorityReplayBufferSampleCpuKernel() override = default;

  // Init kernel from CNode.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  int64_t handle_{-1};
  size_t batch_size_{0};
  std::vector<size_t> schema_;
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferUpdateCpuKernel : public DeprecatedNativeCpuKernelMod {
 public:
  PriorityReplayBufferUpdateCpuKernel() = default;
  ~PriorityReplayBufferUpdateCpuKernel() override = default;

  // Init kernel from CNode.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
  std::vector<int64_t> indices_shape_;
  std::vector<int64_t> priorities_shape_;
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferDestroyCpuKernel : public DeprecatedNativeCpuKernelMod {
 public:
  PriorityReplayBufferDestroyCpuKernel() = default;
  ~PriorityReplayBufferDestroyCpuKernel() override = default;

  // Collect and prepare kernel algorithm parameter.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRIORITY_REPLAY_BUFFER_CPU_KERNEL_H_
