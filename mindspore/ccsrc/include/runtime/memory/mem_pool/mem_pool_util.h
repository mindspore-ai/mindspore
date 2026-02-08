/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_POOL_UTIL_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_POOL_UTIL_H_

#include <atomic>
#include <string>

#include "include/runtime/memory/mem_pool/mem_env.h"
#include "include/backend/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace memory {
namespace mem_pool {
enum class MemType : int {
  kWeight = 0,
  kConstantValue,
  kKernel,
  kGraphOutput,
  kSomas,
  kSomasOutput,
  kGeConst,
  kGeFixed,
  kBatchMemory,
  kContinuousMemory,
  kPyNativeInput = 10,
  kPyNativeOutput,
  kWorkSpace,
  kOther
};

class Lock {
 public:
  inline void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
    }
  }
  inline void unlock() { locked.clear(std::memory_order_release); }

 protected:
  std::atomic_flag locked = ATOMIC_FLAG_INIT;
};

class BACKEND_EXPORT LockGuard {
 public:
  explicit LockGuard(const Lock &lock);
  ~LockGuard();

 private:
  Lock *lock_;
};

BACKEND_EXPORT std::string MemTypeToStr(MemType mem_type);
BACKEND_EXPORT bool IsEnableMemTrack();
BACKEND_EXPORT bool IsNeedProfilieMemoryLog();
BACKEND_EXPORT bool IsMemoryPoolRecycle();

std::string GeneratePath(size_t rank_id, const std::string &file_name, const std::string &suffix);
}  // namespace mem_pool
}  // namespace memory
}  // namespace mindspore
#endif
