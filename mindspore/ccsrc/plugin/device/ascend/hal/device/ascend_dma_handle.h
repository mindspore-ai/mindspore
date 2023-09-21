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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DMA_HANDLE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DMA_HANDLE_H_

#include <memory>
#include <string>
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendDmaHandle {
 public:
  ~AscendDmaHandle();
  static AscendDmaHandle &GetInstance();
  void *GetBuf() const;
  void *GetDargs() const;
  size_t GetSize() const;

 private:
  AscendDmaHandle();
  AscendDmaHandle(const AscendDmaHandle &) = delete;
  AscendDmaHandle &operator=(const AscendDmaHandle &) = delete;
  void InitRuntimeInstance();
  void InitDmaMem();
  void *dargs_{nullptr};
  size_t device_hbm_free_size_{0};
  size_t device_hbm_total_size_{0};
  int p2p_fd_{0};
  void *buf_{nullptr};
  size_t hbm_alloc_size_ = 1 << 30;
  uint32_t device_id_{0};
  KernelRuntime *runtime_instance_{nullptr};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_DMA_HANDLE_H_
