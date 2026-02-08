/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/memory_manager/pin_mem_pool.h"
#include <string>
#include <algorithm>
#include <cstdlib>
#include "utils/log_adapter.h"
#include "utils/distributed_meta.h"
#include "include/utils/offload_context.h"
#include "include/utils/comm_manager.h"

namespace mindspore {
namespace device {
constexpr size_t kMemPoolAlignSize = 512;

PinMemPool::PinMemPool() {
  const auto &offload_context = OffloadContext::GetInstance();
  pinned_mem_ = offload_context->enable_pinned_mem();
}

void *PinMemPool::AllocPinMem(size_t size, uint32_t stream_id) {
  if (!inited_) {
    Init();
  }
  return AllocTensorMem(size, false, false, stream_id);
}

size_t PinMemPool::AllocDeviceMem(size_t alloc_size, DeviceMemPtr *addr) {
  if (alloc_size == 0) {
    MS_LOG(EXCEPTION) << "The memory alloc size is 0.";
  }

#if defined(_WIN32) || defined(_WIN64)
  *addr = malloc(alloc_size);
#else
  if (!pinned_mem_) {
    auto status = posix_memalign(addr, kMemPoolAlignSize, alloc_size);
    if (status != 0) {
      MS_LOG(ERROR) << "The PinMemPool posix_memalign failed, error code is " << status << ".";
    }
  } else {
    PinnedMemAlloc(addr, alloc_size);
  }
#endif
  if (*addr == nullptr) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return 0;
  }
  total_used_memory_ += alloc_size;
  MS_LOG(INFO) << "Current PinMemPool alloc size[" << alloc_size << "], total used size[" << total_used_memory_
               << "], available host mem size [" << max_size_ - total_used_memory_ << "].";
  return alloc_size;
}

void PinMemPool::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (inited_) {
    return;
  }
  const auto &offload_context = OffloadContext::GetInstance();
  auto cpu_mem_size = offload_context->offload_cpu_size();
  if (!offload_context->cpu_size_configured()) {
    auto local_rank_size = DistributedMeta::GetInstance()->local_rank_size();
    if (local_rank_size == 0) {
      MS_LOG(ERROR) << "Local rank size can not be 0, reset to 1.";
      local_rank_size = 1;
    }
    cpu_mem_size = cpu_mem_size / local_rank_size;
  }
  max_size_ = cpu_mem_size;
  SetMemPoolBlockSize(max_size_);
  inited_ = true;
  MS_LOG(INFO) << "PinMemPool init success.";
}

void PinMemPool::SetMemPoolBlockSize(size_t available_pin_mem_size) {
  const auto &offload_context = OffloadContext::GetInstance();
  auto real_block_size = std::min(available_pin_mem_size, offload_context->host_mem_block_size());
  SetMemAllocUintSize(real_block_size);
}

size_t PinMemPool::free_mem_size() { return max_size_ - total_used_memory_; }
}  // namespace device
}  // namespace mindspore
