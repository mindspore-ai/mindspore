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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_HOST_DYNAMIC_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_HOST_DYNAMIC_KERNEL_H_

#include "runtime/device/executor/dynamic_kernel.h"

namespace mindspore {
namespace device {
namespace ascend {
class HostDynamicKernel : public DynamicKernel {
 public:
  HostDynamicKernel(void *stream, const CNodePtr &cnode_ptr) : DynamicKernel(stream, cnode_ptr) {}
  ~HostDynamicKernel() override = default;
  void UpdateArgs() override {}
  void Execute() override = 0;
  void PostExecute() override {}
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_HOST_DYNAMIC_KERNEL_H_
