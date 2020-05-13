/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/data/dataset_init_kernel.h"
#include "kernel/gpu/data/dataset_utils.h"
#include "device/gpu/gpu_buffer_mgr.h"
#include "device/gpu/gpu_memory_allocator.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::device::GpuBufferMgr;

DatasetInitKernel::DatasetInitKernel() : total_bytes_(0) {}

const std::vector<size_t> &DatasetInitKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &DatasetInitKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &DatasetInitKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool DatasetInitKernel::Init(const CNodePtr &kernel_node) {
  queue_name_ = GetAttr<std::string>(kernel_node, "queue_name");
  auto shapes = GetAttr<const std::vector<std::vector<int>>>(kernel_node, "shapes");
  auto types = GetAttr<const std::vector<TypePtr>>(kernel_node, "types");
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "Invalid shapes: " << shapes << ", types: " << types;
  }

  for (size_t i = 0; i < shapes.size(); i++) {
    int unit = UnitSizeInBytes(types[i]->type_id());
    int nums = ElementNums(shapes[i]);
    int bytes = unit * nums;
    shapes_.push_back(bytes);
    total_bytes_ += bytes;
  }
  return true;
}

void DatasetInitKernel::InitSizeLists() { return; }

bool DatasetInitKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &, uintptr_t) {
  void *addr = nullptr;
  size_t len = total_bytes_ * buffer_q_capacity_;

  if (!device::gpu::GPUMemoryAllocator::GetInstance().AllocBufferQueueMem(len, &addr)) {
    MS_LOG(EXCEPTION) << "Memory not enough: failed to allocate GPU buffer queue memory[" << len << "].";
  }

  auto status = GpuBufferMgr::GetInstance().Create(0, queue_name_, addr, shapes_, buffer_q_capacity_);
  if (status) {
    MS_LOG(EXCEPTION) << "Init Dataset Failed. len: " << len << ", status:" << status;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
