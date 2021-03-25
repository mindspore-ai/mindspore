/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LAUNCH_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LAUNCH_KERNEL_H_

#include <vector>
#include <memory>
#include "runtime/device/launch_kernel.h"

namespace mindspore::device::ascend {
class AscendLaunchKernel : public LaunchKernel {
 public:
  explicit AscendLaunchKernel(void *stream) : LaunchKernel(stream) {}
  virtual ~AscendLaunchKernel() = default;

  void FreeDeviceMem(void *addr) override;
  size_t AlignSizeForLaunchKernel(size_t size) override;
  uint8_t *AllocDeviceMem(size_t size) override;
  void KernelSelect(std::shared_ptr<session::KernelGraph> kernel_graph) override;
  void KernelBuild(std::shared_ptr<session::KernelGraph> kernel_graph) override;

  void SetInputAddr(uint8_t *input_addr) override = 0;
  void LaunchOpKernel() override = 0;
  void FreeLaunchDeviceMem() override = 0;
};
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LAUNCH_KERNEL_H_
