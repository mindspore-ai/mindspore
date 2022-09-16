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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_PRIORITY_REPLAY_BUFFER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_PRIORITY_REPLAY_BUFFER_GPU_KERNEL_H_

#include "plugin/device/gpu/kernel/rl/priority_replay_buffer.h"
#include <vector>
#include <memory>
#include "plugin/factory/replay_buffer_factory.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
using gpu::PriorityReplayBuffer;
class PriorityReplayBufferCreateGpuKernel : public NativeGpuKernelMod {
 public:
  PriorityReplayBufferCreateGpuKernel() = default;
  ~PriorityReplayBufferCreateGpuKernel() override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t handle_{-1};
  int64_t *handle_device_{nullptr};
  std::shared_ptr<PriorityReplayBuffer<SumMinTree>> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferPushGpuKernel : public NativeGpuKernelMod {
 public:
  PriorityReplayBufferPushGpuKernel() = default;
  ~PriorityReplayBufferPushGpuKernel() override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t handle_{-1};
  int64_t *handle_device_{nullptr};
  // The API prototype is push(*transitions, priority), and the last `priority` is an optional argument.
  // Default priority is used When the `priority` is not provided.
  size_t num_item_{0};
  bool default_priority_{true};
  std::shared_ptr<PriorityReplayBuffer<SumMinTree>> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferSampleGpuKernel : public NativeGpuKernelMod {
 public:
  PriorityReplayBufferSampleGpuKernel() = default;
  ~PriorityReplayBufferSampleGpuKernel() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t handle_{-1};
  size_t batch_size_{0};
  std::vector<size_t> schema_;
  std::shared_ptr<PriorityReplayBuffer<SumMinTree>> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferUpdateGpuKernel : public NativeGpuKernelMod {
 public:
  PriorityReplayBufferUpdateGpuKernel() = default;
  ~PriorityReplayBufferUpdateGpuKernel() override;

  // Init kernel from CNode.
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t handle_{-1};
  int64_t *handle_device_{nullptr};
  size_t batch_size_{0};
  std::shared_ptr<PriorityReplayBuffer<SumMinTree>> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferDestroyGpuKernel : public NativeGpuKernelMod {
 public:
  PriorityReplayBufferDestroyGpuKernel() = default;
  ~PriorityReplayBufferDestroyGpuKernel() override = default;

  // Init kernel from CNode.
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t handle_{-1};
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, PriorityReplayBufferCreate, PriorityReplayBufferCreateGpuKernel);
MS_REG_GPU_KERNEL(PriorityReplayBufferPush, PriorityReplayBufferPushGpuKernel)
MS_REG_GPU_KERNEL(PriorityReplayBufferSample, PriorityReplayBufferSampleGpuKernel)
MS_REG_GPU_KERNEL(PriorityReplayBufferUpdate, PriorityReplayBufferUpdateGpuKernel)
MS_REG_GPU_KERNEL(PriorityReplayBufferDestroy, PriorityReplayBufferDestroyGpuKernel)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_PRIORITY_REPLAY_BUFFER_GPU_KERNEL_H_
