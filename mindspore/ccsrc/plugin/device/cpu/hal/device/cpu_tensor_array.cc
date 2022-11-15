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

#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include <vector>
#include <string>
#include <memory>
#include "plugin/device/cpu/hal/hardware/cpu_memory_pool.h"

namespace mindspore {
namespace device {
namespace cpu {
void *CPUTensorArray::AllocateMemory(const size_t size) { return CPUMemoryPool::GetInstance().AllocTensorMem(size); }

void CPUTensorArray::ClearMemory(void *addr, const size_t size) {
  if (memset_s(addr, size, 0, size) != EOK) {
    MS_LOG(EXCEPTION) << "Failed to clear memory.";
  }
}

void CPUTensorArray::FreeMemory(const DeviceMemPtr addr) { CPUMemoryPool::GetInstance().FreeTensorMem(addr); }

void CPUTensorsQueue::CopyTensor(const mindspore::kernel::AddressPtr &dst, const mindspore::kernel::AddressPtr &src) {
  if (dst->size != src->size) {
    MS_LOG(EXCEPTION) << "For TensorsQueue Put/Get function, each tensor in element should have the same size, but get "
                      << src->size << " not equal to dst " << dst->size;
  }
  if (memcpy_s(dst->addr, dst->size, src->addr, src->size) != EOK) {
    MS_LOG(EXCEPTION) << "CopyTensor failed";
  }
}
void *CPUTensorsQueue::AllocateMemory(const size_t size) { return CPUMemoryPool::GetInstance().AllocTensorMem(size); }

void CPUTensorsQueue::ClearMemory(void *addr, const size_t size) {
  if (memset_s(addr, size, 0, size) != EOK) {
    MS_LOG(EXCEPTION) << "Failed to clear memory.";
  }
}

void CPUTensorsQueue::FreeMemory(const DeviceMemPtr addr) { CPUMemoryPool::GetInstance().FreeTensorMem(addr); }
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
