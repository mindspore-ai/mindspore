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

#include "plugin/device/gpu/kernel/rl/priority_replay_buffer.h"

#include <curand_kernel.h>
#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include "kernel/kernel.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"

namespace mindspore {
namespace kernel {
namespace gpu {
constexpr float kMinPriority = 1e-7;
constexpr size_t kNumSubNode = 2;

PriorityReplayBuffer::PriorityReplayBuffer(const uint64_t &seed, const float &alpha, const size_t &capacity,
                                           const std::vector<size_t> &schema) {
  alpha_ = alpha;
  schema_ = schema;
  seed_ = seed;
  capacity_ = capacity;

  // FIFO used for storing transitions
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  for (const auto &size : schema) {
    fifo_replay_buffer_.emplace_back(static_cast<uint8_t *>(allocator.AllocTensorMem(size * capacity)));
  }

  // The sum tree used for keeping priority.
  max_priority_ = static_cast<float *>(allocator.AllocTensorMem(sizeof(float)));

  capacity_pow_two_ = 1;
  while (capacity_pow_two_ < capacity) {
    capacity_pow_two_ *= kNumSubNode;
  }
  sum_tree_ = static_cast<SumTree *>(allocator.AllocTensorMem(capacity_pow_two_ * sizeof(SumTree) * kNumSubNode));
  // Set initial segment info for all element.
  SumTreeInit(sum_tree_, max_priority_, capacity_pow_two_, nullptr);
}

PriorityReplayBuffer::~PriorityReplayBuffer() {
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  if (rand_state_) {
    allocator.FreeTensorMem(rand_state_);
    rand_state_ = nullptr;
  }

  for (auto item : fifo_replay_buffer_) {
    allocator.FreeTensorMem(item);
  }

  allocator.FreeTensorMem(sum_tree_);
  allocator.FreeTensorMem(max_priority_);
}

bool PriorityReplayBuffer::Push(const std::vector<AddressPtr> &transition, float *priority, cudaStream_t stream) {
  // Head point to the latest item.
  head_ = head_ >= capacity_ ? 0 : head_ + 1;
  valid_size_ = valid_size_ >= capacity_ ? capacity_ : valid_size_ + 1;

  // Copy transition to FIFO.
  for (size_t i = 0; i < transition.size(); i++) {
    size_t offset = head_ * schema_[i];
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(fifo_replay_buffer_[i] + offset, transition[i]->addr, schema_[i],
                                                      cudaMemcpyDeviceToDevice, stream),
                                      "cudaMemcpyAsync failed.");
  }

  // Set max priority for the newest transition.
  SumTreePush(sum_tree_, alpha_, head_, capacity_pow_two_, priority, max_priority_, stream);
  return true;
}

bool PriorityReplayBuffer::Sample(const size_t &batch_size, float *beta, size_t *indices, float *weights,
                                  const std::vector<AddressPtr> &transition, cudaStream_t stream) {
  MS_EXCEPTION_IF_ZERO("batch size", batch_size);

  // Init random state for sampling.
  if (!rand_state_) {
    auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
    rand_state_ = static_cast<curandState *>(allocator.AllocTensorMem(sizeof(curandState) * batch_size));
    InitRandState(batch_size, seed_, rand_state_, stream);
  }

  SumTreeSample(sum_tree_, rand_state_, capacity_pow_two_, beta, batch_size, indices, weights, stream);

  for (size_t i = 0; i < schema_.size(); i++) {
    auto output_addr = static_cast<uint8_t *>(transition[i]->addr);
    FifoSlice(fifo_replay_buffer_[i], indices, output_addr, batch_size, schema_[i], stream);
  }

  return true;
}

bool PriorityReplayBuffer::UpdatePriorities(size_t *indices, float *priorities, const size_t &batch_size,
                                            cudaStream_t stream) {
  SumTreeUpdate(sum_tree_, capacity_pow_two_, alpha_, max_priority_, indices, priorities, batch_size, stream);
  return true;
}
}  // namespace gpu
}  // namespace kernel
}  // namespace mindspore
