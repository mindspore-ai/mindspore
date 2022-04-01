/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_TENSORS_QUEUE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_TENSORS_QUEUE_GPU_KERNEL_H_

#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/rl/tensors_queue_gpu_base.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class TensorsQueueCreateKernelMod : public TensorsQueueBaseMod {
 public:
  TensorsQueueCreateKernelMod();
  ~TensorsQueueCreateKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;

 private:
  std::string name_;
  int64_t size_;
  int64_t elements_num_;
  std::vector<std::vector<int64_t>> shapes_;
  TypePtr type_;
};

class TensorsQueuePutKernelMod : public TensorsQueueBaseMod {
 public:
  TensorsQueuePutKernelMod();
  ~TensorsQueuePutKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;

 private:
  int64_t elements_num_;
  TypeId type_;
};

class TensorsQueueGetKernelMod : public TensorsQueueBaseMod {
 public:
  TensorsQueueGetKernelMod();
  ~TensorsQueueGetKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;

 private:
  int64_t elements_num_;
  bool pop_after_get_;
};

class TensorsQueueClearKernelMod : public TensorsQueueBaseMod {
 public:
  TensorsQueueClearKernelMod();
  ~TensorsQueueClearKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;
};

class TensorsQueueCloseKernelMod : public TensorsQueueBaseMod {
 public:
  TensorsQueueCloseKernelMod();
  ~TensorsQueueCloseKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;
};

class TensorsQueueSizeKernelMod : public TensorsQueueBaseMod {
 public:
  TensorsQueueSizeKernelMod();
  ~TensorsQueueSizeKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;
};

MS_REG_GPU_KERNEL(TensorsQueueCreate, TensorsQueueCreateKernelMod)
MS_REG_GPU_KERNEL(TensorsQueuePut, TensorsQueuePutKernelMod)
MS_REG_GPU_KERNEL(TensorsQueueGet, TensorsQueueGetKernelMod)
MS_REG_GPU_KERNEL(TensorsQueueClear, TensorsQueueClearKernelMod)
MS_REG_GPU_KERNEL(TensorsQueueClose, TensorsQueueCloseKernelMod)
MS_REG_GPU_KERNEL(TensorsQueueSize, TensorsQueueSizeKernelMod)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_TENSORS_QUEUE_GPU_KERNEL_H_
