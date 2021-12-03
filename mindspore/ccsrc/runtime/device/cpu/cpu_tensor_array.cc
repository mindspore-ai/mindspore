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

#include "runtime/device/cpu/cpu_tensor_array.h"
#include <vector>
#include <string>
#include <memory>
#include "runtime/hardware/cpu/cpu_memory_pool.h"

namespace mindspore {
namespace device {
namespace cpu {
void *CPUTensorArray::CreateMemory(const size_t size) { return CPUMemoryPool::GetInstance().AllocTensorMem(size); }

void CPUTensorArray::ClearMemory(void *addr, const size_t size) { (void)memset_s(addr, size, 0, size); }

void CPUTensorArray::ReleaseMemory(void *addr) { CPUMemoryPool::GetInstance().FreeTensorMem(addr); }
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
