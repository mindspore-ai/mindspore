/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/data/dataset_init_kernel.h"
#include "plugin/device/gpu/kernel/data/dataset_utils.h"
#include "kernel/common_utils.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::device::DataQueueMgr;

DatasetInitKernelMod::DatasetInitKernelMod() : total_bytes_(0) {}

bool DatasetInitKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  queue_name_ = GetAttr<std::string>(kernel_node, "queue_name");
  std::vector<std::vector<int>> shapes;
  std::vector<TypePtr> types;
  GetShapeAndType(kernel_node, &shapes, &types);
  for (auto item : types) {
    MS_EXCEPTION_IF_NULL(item);
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

void DatasetInitKernelMod::InitSizeLists() { return; }

bool DatasetInitKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                  const std::vector<AddressPtr> &, void *) {
  auto status = DataQueueMgr::GetInstance().Create(queue_name_, shapes_, buffer_q_capacity_);
  if (status != device::DataQueueStatus::SUCCESS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', init Dataset Failed, status:" << status;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
