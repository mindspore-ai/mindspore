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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_RESERVOIR_REPLAY_BUFFER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_RESERVOIR_REPLAY_BUFFER_H_

#include <vector>
#include "common/kernel_base.h"
#include "replay_buffer/reservoir_replay_buffer.h"

namespace aicpu {
class ReservoirReplayBufferCreate : public KernelBase {
 public:
  ReservoirReplayBufferCreate() : KernelBase("ReservoirReplayBufferCreate") {}
  ~ReservoirReplayBufferCreate() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t capacity_{0};
  int64_t seed_{0};
  std::vector<size_t> schema_;
};

class ReservoirReplayBufferPush : public KernelBase {
 public:
  ReservoirReplayBufferPush() : KernelBase("ReservoirReplayBufferPush") {}
  ~ReservoirReplayBufferPush() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t handle_{-1};
  std::vector<AddressPtr> inputs_;
};

class ReservoirReplayBufferSample : public KernelBase {
 public:
  ReservoirReplayBufferSample() : KernelBase("ReservoirReplayBufferSample") {}
  ~ReservoirReplayBufferSample() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t handle_{-1};
  int64_t batch_size_{0};
  std::vector<size_t> schema_;
};

class ReservoirReplayBufferDestroy : public KernelBase {
 public:
  ReservoirReplayBufferDestroy() : KernelBase("ReservoirReplayBufferDestroy") {}
  ~ReservoirReplayBufferDestroy() = default;

 protected:
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;

 private:
  int64_t handle_{-1};
};
}  // namespace aicpu
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_RESERVOIR_REPLAY_BUFFER_H_
