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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_PRIORITY_REPLAY_BUFFER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_PRIORITY_REPLAY_BUFFER_H_

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>
#include "kernel/kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/rl/priority_replay_buffer.cuh"

namespace mindspore {
namespace kernel {
namespace gpu {
// PriorityReplayBuffer is experience container used in Deep Q-Networks.
// The algorithm is proposed in `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`.
// Same as the normal replay buffer, it lets the reinforcement learning agents remember and reuse experiences from the
// past. Besides, it replays important transitions more frequently and improve sample effciency.
class PriorityReplayBuffer {
 public:
  // Construct a fixed-length priority replay buffer.
  PriorityReplayBuffer(const uint64_t &seed, const float &alpha, const size_t &capacity,
                       const std::vector<size_t> &schema);
  ~PriorityReplayBuffer();

  // Push an experience transition to the buffer which will be given the highest priority.
  bool Push(const std::vector<AddressPtr> &transition, float *priority, cudaStream_t stream);

  // Sample a batch transitions with indices and bias correction weights.
  bool Sample(const size_t &batch_size, float *beta, size_t *indices, float *weights,
              const std::vector<AddressPtr> &transition, cudaStream_t stream);

  // Update experience transitions priorities.
  bool UpdatePriorities(size_t *indices, float *priorities, const size_t &batch_size, cudaStream_t stream);

 private:
  float alpha_{1.};

  std::vector<size_t> schema_;
  uint64_t seed_{42};
  curandState *rand_state_{nullptr};

  size_t capacity_{0};
  size_t valid_size_{0};
  size_t head_{-1UL};
  std::vector<uint8_t *> fifo_replay_buffer_;

  size_t capacity_pow_two_{0};
  float *max_priority_{nullptr};
  SumTree *sum_tree_{nullptr};
};
}  // namespace gpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_PRIORITY_REPLAY_BUFFER_H_
