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

#include "plugin/device/gpu/kernel/rl/reservoir_replay_buffer.h"

#include <curand_kernel.h>
#include <vector>
#include <memory>
#include <algorithm>
#include "kernel/kernel.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "plugin/device/gpu/kernel/cuda_impl/rl/rl_buffer_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/rl/priority_replay_buffer.cuh"

namespace mindspore {
namespace kernel {
namespace gpu {
ReservoirReplayBuffer::ReservoirReplayBuffer(const uint64_t &seed, const size_t &capacity,
                                             const std::vector<size_t> &schema) {
  capacity_ = capacity;

  // Init random generator.
  seed_ = seed;
  generator_.seed(seed_);

  // Allocate device memory.
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  schema_ = schema;
  for (const auto &size : schema) {
    fifo_replay_buffer_.emplace_back(static_cast<uint8_t *>(allocator.AllocTensorMem(size * capacity)));
  }
}

ReservoirReplayBuffer::~ReservoirReplayBuffer() {
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  if (indices_) {
    allocator.FreeTensorMem(indices_);
    indices_ = nullptr;
  }

  if (rand_state_) {
    allocator.FreeTensorMem(rand_state_);
    rand_state_ = nullptr;
  }

  for (auto item : fifo_replay_buffer_) {
    allocator.FreeTensorMem(item);
  }
}

bool ReservoirReplayBuffer::Insert(const size_t &pos, const std::vector<AddressPtr> &transition, cudaStream_t stream) {
  for (size_t i = 0; i < transition.size(); i++) {
    size_t offset = pos * schema_[i];
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(fifo_replay_buffer_[i] + offset, transition[i]->addr, schema_[i],
                                                      cudaMemcpyDeviceToDevice, stream),
                                      "cudaMemcpyAsync failed.");
  }
  return true;
}

bool ReservoirReplayBuffer::Push(const std::vector<AddressPtr> &transition, cudaStream_t stream) {
  // The buffer is not full: Push the transition at end of buffer.
  if (total_ < capacity_) {
    auto ret = Insert(total_, transition, stream);
    total_++;
    return ret;
  }

  // The buffer is full: Random discard this sample or replace the an old one.
  auto replace_threthold = static_cast<float>(capacity_) / static_cast<float>(total_);
  std::uniform_real_distribution<float> keep_dist(0, 1);
  auto prob = keep_dist(generator_);
  if (prob < replace_threthold) {
    total_++;
    std::uniform_int_distribution<size_t> pos_dist(0, capacity_ - 1);
    size_t pos = pos_dist(generator_);
    return Insert(pos, transition, stream);
  }

  return true;
}

bool ReservoirReplayBuffer::Sample(const size_t &batch_size, const std::vector<AddressPtr> &transition,
                                   cudaStream_t stream) {
  cudaError_t status = cudaErrorNotReady;
  if (!rand_state_) {
    auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
    rand_state_ = static_cast<curandState *>(allocator.AllocTensorMem(sizeof(curandState) * batch_size));
    status = RandInit(batch_size, seed_, rand_state_, stream);
    CHECK_CUDA_STATUS(status, "RandInit called by Sample");
  }

  if (!indices_) {
    auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
    indices_ = static_cast<size_t *>(allocator.AllocTensorMem(sizeof(size_t) * batch_size));
  }

  size_t valid_size = std::min(total_, capacity_);
  status = RandomGenUniform(batch_size, rand_state_, valid_size, indices_, stream);
  CHECK_CUDA_STATUS(status, "RandomGenUniform called by Sample");

  for (size_t i = 0; i < schema_.size(); i++) {
    auto output_addr = static_cast<uint8_t *>(transition[i]->addr);
    status = FifoSlice(fifo_replay_buffer_[i], indices_, output_addr, batch_size, schema_[i], stream);
    CHECK_CUDA_STATUS(status, "FifoSlice called by Sample");
  }

  return true;
}
}  // namespace gpu
}  // namespace kernel
}  // namespace mindspore
