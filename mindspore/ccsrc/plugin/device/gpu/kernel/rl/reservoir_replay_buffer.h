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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_RESERVOIR_REPLAY_BUFFER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_RESERVOIR_REPLAY_BUFFER_H_

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>
#include <random>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
namespace gpu {
class ReservoirReplayBuffer {
 public:
  // Construct a fixed-length reservoir replay buffer.
  ReservoirReplayBuffer(const uint64_t &seed, const size_t &capacity, const std::vector<size_t> &schema);
  ~ReservoirReplayBuffer();

  // Push an experience transition to the buffer which will be given the highest reservoir.
  bool Push(const std::vector<AddressPtr> &transition, cudaStream_t stream);

  // Sample a batch transitions with indices and bias correction weights.
  bool Sample(const size_t &batch_size, const std::vector<AddressPtr> &transition, cudaStream_t stream);

 private:
  bool Insert(const size_t &pos, const std::vector<AddressPtr> &transition, cudaStream_t stream);

  // Random generator
  std::default_random_engine generator_;
  curandState *rand_state_{nullptr};

  uint64_t seed_{42};
  size_t capacity_{0};
  size_t total_{0};
  size_t *indices_{nullptr};
  std::vector<size_t> schema_;
  std::vector<uint8_t *> fifo_replay_buffer_;
};
}  // namespace gpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_RL_RESERVOIR_REPLAY_BUFFER_H_
