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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_RTS_PROFILING_RTS_DYNAMIC_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_RTS_PROFILING_RTS_DYNAMIC_KERNEL_H_

#include "runtime/device/executor/dynamic_kernel.h"

namespace mindspore {
namespace device {
namespace ascend {
class ProfilingRtsDynamicKernel : public DynamicKernel {
 public:
  ProfilingRtsDynamicKernel(void *stream, const CNodePtr &cnode_ptr, uint64_t log_id, bool notify, uint32_t flags)
      : DynamicKernel(stream, cnode_ptr), log_id_(log_id), notify_(notify), flags_(flags) {}
  ~ProfilingRtsDynamicKernel() override = default;

  void UpdateArgs() override {}
  void Execute() override;
  void PostExecute() override {}

 private:
  uint64_t log_id_;
  bool notify_;
  uint32_t flags_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_RTS_PROFILING_RTS_DYNAMIC_KERNEL_H_
