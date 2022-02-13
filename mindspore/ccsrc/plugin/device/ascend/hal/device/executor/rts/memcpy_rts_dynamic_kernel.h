/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_RTS_MEMCPY_RTS_DYNAMIC_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_RTS_MEMCPY_RTS_DYNAMIC_KERNEL_H_

#include "runtime/device/executor/dynamic_kernel.h"

namespace mindspore {
namespace device {
namespace ascend {
class MemcpyRtsDynamicKernel : public DynamicKernel {
 public:
  MemcpyRtsDynamicKernel(void *stream, const CNodePtr &cnode_ptr, void *dst, uint32_t dest_max, void *src,
                         uint32_t count)
      : DynamicKernel(stream, cnode_ptr), dst_(dst), dest_max_(dest_max), src_(src), count_(count) {}
  ~MemcpyRtsDynamicKernel() override = default;

  void UpdateArgs() override {}
  void Execute() override;
  void PostExecute() override {}

 private:
  void *dst_;
  uint32_t dest_max_;
  void *src_;
  uint32_t count_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_RTS_MEMCPY_RTS_DYNAMIC_KERNEL_H_
