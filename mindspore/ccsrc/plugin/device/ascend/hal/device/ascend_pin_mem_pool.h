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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_PIN_MEM_POOL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_PIN_MEM_POOL_H_

#include "runtime/device/gsm/pin_mem_pool.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendPinMemPool : public PinMemPool {
 public:
  ~AscendPinMemPool() = default;
  static AscendPinMemPool &GetInstance();

 private:
  AscendPinMemPool();
  AscendPinMemPool(const AscendPinMemPool &) = delete;
  AscendPinMemPool &operator=(const AscendPinMemPool &) = delete;
  bool FreeDeviceMem(const DeviceMemPtr &addr) override;
  void PinnedMemAlloc(DeviceMemPtr *addr, size_t alloc_size) override;
  AscendKernelRuntime *runtime_instance_{nullptr};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_PIN_MEM_POOL_H_
