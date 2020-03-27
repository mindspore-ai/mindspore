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
#include "device/gpu/gpu_buffer_mgr.h"
#include "device/gpu/gpu_memory_allocator.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::device::GpuBufferMgr;

DatasetInitKernel::DatasetInitKernel() : feature_size_(0), label_size_(0) {}

const std::vector<size_t> &DatasetInitKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &DatasetInitKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &DatasetInitKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

size_t DatasetInitKernel::TensorSize(std::vector<int> &shape) const {
  if (shape.size() == 0) {
    return 0;
  }

  int size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }

  return IntToSize(size);
}

bool DatasetInitKernel::Init(const CNodePtr &kernel_node) {
  queue_name_ = GetAttr<std::string>(kernel_node, "queue_name");
  auto shapes = GetAttr<const std::vector<std::vector<int>>>(kernel_node, "shapes");
  auto data_num = shapes.size();
  if (data_num != 2) {
    MS_LOG(EXCEPTION) << "Invalid Shapes " << data_num;
  }

  auto &feature_Shapes = shapes[0];
  auto size = TensorSize(feature_Shapes);
  feature_size_ = size * sizeof(float);

  auto types = GetAttr<const std::vector<TypePtr>>(kernel_node, "types");
  if ((types[1]->type_id() != kNumberTypeInt32) && (types[1]->type_id() != kNumberTypeInt64)) {
    MS_LOG(EXCEPTION) << "Invalid types " << types[1]->type_id();
  }

  size_t label_unit = (types[1]->type_id() == kNumberTypeInt32) ? sizeof(int32_t) : sizeof(int64_t);
  size = TensorSize(shapes[1]);
  label_size_ = size * label_unit;
  return true;
}

void DatasetInitKernel::InitSizeLists() { return; }

bool DatasetInitKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &, uintptr_t) {
  void *addr = nullptr;
  size_t len = (feature_size_ + label_size_) * buffer_q_capacity_;

  if (!device::gpu::GPUMemoryAllocator::GetInstance().AllocBufferQueueMem(len, &addr)) {
    MS_LOG(EXCEPTION) << "Memory not enough: failed to allocate GPU buffer queue memory[" << len << "].";
  }

  auto status =
    GpuBufferMgr::GetInstance().Create(0, queue_name_, addr, feature_size_, label_size_, buffer_q_capacity_);
  if (status) {
    MS_LOG(EXCEPTION) << "Init Dataset Failed: " << queue_name_ << ", " << feature_size_ << ", " << label_size_ << ", "
                      << status;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
