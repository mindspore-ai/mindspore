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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_GMEM_ADAPTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_GMEM_ADAPTER_H_

#include <atomic>
#include <memory>

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "utils/dlopen_macro.h"
#include "utils/hash_map.h"
#include "utils/log_adapter.h"

#include "include/backend/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
namespace ascend {
#define CONCAT(l, r) l##r
// Function Object definition marco.
#define LIB_FUNC(func_name) CONCAT(func_name, FunObj)
// Function definition marco, and then can ues `LIB_FUNC(func_name)`.
#define DEFINE_LIB_METHOD(func_name, ...) ORIGIN_METHOD(func_name, __VA_ARGS__)

// GMem mem free eager function name. Need to use origin name when export symbol from lib.
#define GMEM_FREE_EAGER gmemFreeEager
// Definition for GMem lib function : GMEM_FREE_EAGER.
DEFINE_LIB_METHOD(GMEM_FREE_EAGER, size_t, uint64_t, size_t, void *);

struct CallbackThread;
using CallbackThreadPtr = std::shared_ptr<CallbackThread>;

class AscendGmemAdapter {
 public:
  static AscendGmemAdapter &GetInstance() {
    static AscendGmemAdapter instance{};
    return instance;
  }

  AscendGmemAdapter() { LoadGMemLib(); }
  ~AscendGmemAdapter() {
#ifdef WITH_BACKEND
    for (auto iter = callback_map_.begin(); iter != callback_map_.end();) {
      rtStream_t stream = iter->first;
      iter++;
      RemoveCallbackThread(stream);
    }
#endif
    UnloadGMemLib();
  }

 public:
  const size_t GetRoundUpAlignSize(size_t input_size) const;
  const size_t GetRoundDownAlignSize(size_t input_size) const;

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) const;
  size_t EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) const;

  bool AddCallbackThread(rtStream_t stream);
  bool RemoveCallbackThread(rtStream_t stream);

  uint8_t *MmapMemory(size_t size, void *addr) const;
  bool MunmapMemory(void *addr, const size_t size) const;

  inline const bool is_eager_free_enabled() const { return is_eager_free_enabled_; }

 private:
  void LoadGMemLib() noexcept;
  void UnloadGMemLib() noexcept;

  bool is_eager_free_enabled_{false};
  void *gmem_handle_{nullptr};
  // Function for eager free.
  LIB_FUNC(GMEM_FREE_EAGER) free_eager_;
  // Map for call back threads.
  mindspore::HashMap<rtStream_t, CallbackThreadPtr> callback_map_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif
