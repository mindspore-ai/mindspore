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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_PIN_MEM_POOL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_PIN_MEM_POOL_H_

#include <memory>
#include "include/backend/visible.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
class BACKEND_EXPORT PinMemPool : public DynamicMemPoolBestFit {
 public:
  ~PinMemPool() = default;
  virtual void PinnedMemAlloc(DeviceMemPtr *addr, size_t size) = 0;

 protected:
  PinMemPool();
  size_t free_mem_size() override;
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override;
  size_t max_size_{0};
  size_t total_used_memory_{0};
  bool pinned_mem_{false};
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_PIN_MEM_POOL_H_
