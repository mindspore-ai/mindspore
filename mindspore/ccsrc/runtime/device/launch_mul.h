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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LAUNCH_MUL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LAUNCH_MUL_H_

#include <vector>
#include <memory>
#include "include/backend/kernel_graph.h"
#include "kernel/kernel.h"

namespace mindspore::device {
class BACKEND_EXPORT LaunchMul {
 public:
  LaunchMul(TypeId dtype, size_t total_size)
      : dtype_(dtype),
        total_size_(total_size),
        input1_addr_(nullptr),
        input2_addr_(nullptr),
        input2_value_(0),
        mul_graph_(nullptr) {}
  virtual ~LaunchMul() = default;

  virtual void FreeDeviceMem(void *addr) = 0;
  virtual size_t AlignSizeForLaunchKernel(size_t size) = 0;
  virtual uint8_t *AllocDeviceMem(size_t size) = 0;
  virtual void KernelSelect(const std::shared_ptr<session::KernelGraph> &kernel_graph) = 0;
  virtual void KernelBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) = 0;
  virtual void CopyHostMemToDevice(size_t origin_size, size_t dst_size) = 0;

  std::shared_ptr<session::KernelGraph> ObtainMulKernelGraph() const;
  kernel::KernelMod *ObtainLaunchMulKernelMod();
  void ObtainMulInputsAddr();
  void FreeInputDeviceMemory();

 protected:
  TypeId dtype_;
  size_t total_size_;
  uint8_t *input1_addr_;
  uint8_t *input2_addr_;
  float input2_value_;
  std::shared_ptr<session::KernelGraph> mul_graph_;
  std::vector<uint8_t *> inputs_addr_;
};
}  // namespace mindspore::device

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LAUNCH_MUL_H_
