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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LAUNCH_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LAUNCH_KERNEL_H_

#include <vector>
#include <memory>
#include "kernel/kernel.h"
#include "include/backend/kernel_graph.h"

namespace mindspore::device {
class BACKEND_EXPORT LaunchKernel {
 public:
  explicit LaunchKernel(void *stream) : stream_(stream), kernel_mod_(nullptr) {}
  virtual ~LaunchKernel() = default;

  std::vector<uint8_t *> GetKernelOutputAddr() { return outputs_addr_; }
  void LaunchSingleKernel(const std::vector<uint8_t *> &inputs_addr);
  void FreeOutputAndWorkspaceDeviceMem();

  virtual void FreeDeviceMem(void *addr) = 0;
  virtual size_t AlignSizeForLaunchKernel(size_t size) = 0;
  virtual uint8_t *AllocDeviceMem(size_t size) = 0;
  virtual void KernelSelect(const std::shared_ptr<session::KernelGraph> &kernel_graph) = 0;
  virtual void KernelBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) = 0;

  virtual void SetInputAddr(uint8_t *input_addr) = 0;
  virtual void LaunchOpKernel() = 0;
  virtual void FreeLaunchDeviceMem() = 0;

 protected:
  void *stream_;
  kernel::KernelMod *kernel_mod_;
  std::vector<uint8_t *> outputs_addr_;
  std::vector<uint8_t *> workspaces_addr_;

  std::vector<kernel::AddressPtr> ObtainKernelAddress(const std::vector<size_t> &list, std::vector<uint8_t *> *addr);
  std::vector<kernel::AddressPtr> ObtainKernelInputs(const std::vector<size_t> &inputs_list,
                                                     const std::vector<uint8_t *> &inputs_addr);
  std::vector<kernel::AddressPtr> ObtainKernelOutputs(const std::vector<size_t> &outputs_list);
  std::vector<kernel::AddressPtr> ObtainKernelWorkspaces(const std::vector<size_t> &workspaces_list);
};
}  // namespace mindspore::device

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LAUNCH_KERNEL_H_
