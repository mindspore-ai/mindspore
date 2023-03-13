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

#include "runtime/device/gsm/pin_mem_pool.h"
#include <algorithm>
#include <cstdlib>
#include "utils/log_adapter.h"
#include "include/common/utils/offload_context.h"

namespace mindspore {
namespace device {
namespace {
constexpr size_t kMemPoolAlignSize = 512;
}  // namespace
PinMemPool::PinMemPool() {
  const auto &offload_context = OffloadContext::GetInstance();
  max_size_ = offload_context->offload_ddr_size();
  pinned_mem_ = offload_context->enable_pinned_mem();
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

size_t PinMemPool::free_mem_size() { return max_size_ - total_used_memory_; }
}  // namespace device
}  // namespace mindspore
