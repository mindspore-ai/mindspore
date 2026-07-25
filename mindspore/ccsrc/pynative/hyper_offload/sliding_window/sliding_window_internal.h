/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SLIDING_WINDOW_INTERNAL_H_
#define MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SLIDING_WINDOW_INTERNAL_H_

#include <cstdint>
#include <string>
#include "device_address/device_address.h"
#include "include/runtime/memory/mem_pool/dynamic_mem_pool.h"
#include "ir/dtype/type_id.h"
#include "ir/tensor.h"

namespace mindspore {
namespace utils {
namespace sliding_window {

using TensorPtr = tensor::TensorPtr;
using TensorId = uint64_t;

// --- Ascend memory-pool helpers (defined in device_space_manager.cc) ---
size_t AscendPoolIdleSize();
size_t AscendPoolExpandableSize();
size_t AscendPoolEventHeldSize();
void SyncAscendPoolEvents();
bool IsNotEventUsedMemory(device::DeviceAddress *address);
void PrintAscendMemStatistics(const std::string &location);

// Align size to match the memory pool's actual allocation granularity.
// Unlike DynamicMemPool::AlignMemorySize (which returns kDynamicMemAlignSize for 0),
// returns 0 for zero-size inputs since a zero-byte tensor doesn't consume pool memory.
inline size_t AlignToPoolSize(size_t size) {
  if (size == 0) {
    return 0;
  }
  return ((size + device::kDynamicMemAlignSize - 1) / device::kDynamicMemAlignSize) * device::kDynamicMemAlignSize;
}

// Check whether a tensor can be safely offloaded via aclrtMemcpyAsync.
// Non-contiguous tensors or tensors with storage_offset fall back to aclnnInplaceCopy (D2D kernel)
// inside InplaceCopyAscendCustomize, which cannot handle cross-device (device→host) copies.
// Bool tensors may also trigger DMA alignment errors on Ascend.
inline bool CanOffloadTensor(const TensorPtr &tensor) {
  if (tensor == nullptr) return false;
  if (tensor->data_type() == kNumberTypeBool) return false;
  if (!tensor->is_contiguous()) return false;
  if (tensor->storage_offset() != 0) return false;
  if (tensor->storage_info() != nullptr) return false;
  if (tensor->is_parameter()) return false;
  return true;
}

}  // namespace sliding_window
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SLIDING_WINDOW_INTERNAL_H_
