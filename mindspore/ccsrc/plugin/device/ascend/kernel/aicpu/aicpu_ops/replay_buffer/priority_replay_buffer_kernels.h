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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_PRIORITY_REPLAY_BUFFER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_PRIORITY_REPLAY_BUFFER_H_

#include <vector>
#include "common/kernel_base.h"
#include "replay_buffer/priority_replay_buffer.h"

namespace aicpu {
class PriorityReplayBufferCreate : public KernelBase {
 public:
  PriorityReplayBufferCreate() : KernelBase("PriorityReplayBufferCreate") {}
  ~PriorityReplayBufferCreate() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t capacity_{0};
  float alpha_{1.};
  float beta_{1.};
  int64_t seed_{0};
  std::vector<size_t> schema_;
};

class PriorityReplayBufferPush : public KernelBase {
 public:
  PriorityReplayBufferPush() : KernelBase("PriorityReplayBufferPush") {}
  ~PriorityReplayBufferPush() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t handle_{-1};
  std::vector<AddressPtr> inputs_;
};

class PriorityReplayBufferSample : public KernelBase {
 public:
  PriorityReplayBufferSample() : KernelBase("PriorityReplayBufferSample") {}
  ~PriorityReplayBufferSample() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t handle_{-1};
  size_t batch_size_{0};
  std::vector<size_t> schema_;
};

class PriorityReplayBufferUpdate : public KernelBase {
 public:
  PriorityReplayBufferUpdate() : KernelBase("PriorityReplayBufferUpdate") {}
  ~PriorityReplayBufferUpdate() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t handle_{-1};
  size_t batch_size_{0};
  std::vector<AddressPtr> inputs_;
};

class PriorityReplayBufferDestroy : public KernelBase {
 public:
  PriorityReplayBufferDestroy() : KernelBase("PriorityReplayBufferDestroy") {}
  ~PriorityReplayBufferDestroy() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t handle_{-1};
};
}  // namespace aicpu
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_PRIORITY_REPLAY_BUFFER_H_
