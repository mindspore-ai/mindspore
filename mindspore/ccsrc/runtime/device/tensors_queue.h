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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSORS_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSORS_QUEUE_H_

#include <vector>
#include <queue>
#include <string>
#include <memory>
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"

namespace mindspore {
namespace device {
class BACKEND_EXPORT TensorsQueue {
 public:
  // Base TensorsQueue. Constructed by name, dtype, size, elements_num and shapes.
  TensorsQueue(const string &name, const TypePtr &dtype, const int64_t size, const int64_t elements_num,
               const std::vector<std::vector<int64_t>> &shapes)
      : name_(name), dtype_(dtype), shapes_(shapes), size_(size), elements_num_(elements_num) {}
  virtual ~TensorsQueue() = default;
  virtual void CreateTensorsQueue();

  // These three function (FreeMemory, AllocateMemory and ClearMemory) are related with devices.
  // These should be achieved with different devices.
  virtual void FreeMemory(const DeviceMemPtr addr) = 0;
  virtual void *AllocateMemory(const size_t size) = 0;
  virtual void ClearMemory(void *addr, const size_t size) = 0;

  // When memory operations are involved, we need to determine whether to use streams according to the device.
  virtual bool Put(const mindspore::kernel::AddressPtrList &dev_addr);
  virtual bool Put(const mindspore::kernel::AddressPtrList &dev_addr, void *stream);
  virtual void CopyTensor(const mindspore::kernel::AddressPtr &dst, const mindspore::kernel::AddressPtr &src);
  virtual void CopyTensor(const mindspore::kernel::AddressPtr &dst, const mindspore::kernel::AddressPtr &src,
                          void *stream);
  virtual bool Get(const mindspore::kernel::AddressPtrList &dev_addr, const bool &pop_after_get);
  virtual bool Get(const mindspore::kernel::AddressPtrList &dev_addr, const bool &pop_after_get, void *stream);

  // Common functions for TensorsQueue which are device independent.
  virtual void Clear();
  virtual void Free();
  virtual size_t AvailableSize();
  virtual bool IsFull();
  virtual bool IsEmpty();

 protected:
  std::string name_;
  TypePtr dtype_;
  std::vector<std::vector<int64_t>> shapes_;
  int64_t size_;
  int64_t elements_num_;

 private:
  // Using a vector of address list to store the tensors.
  // Using to cursors to simulate the behavior of circular queue.
  std::vector<mindspore::kernel::AddressPtrList> tensors_q_;
  size_t front_ = 0;
  size_t rear_ = 0;
};
using TensorsQueuePtr = std::shared_ptr<TensorsQueue>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSORS_QUEUE_H_
