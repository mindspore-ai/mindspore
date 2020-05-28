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

#include "kernel/rts/assign.h"

#include <memory>

#include "runtime/mem.h"
#include "common/utils.h"

using ge::model_runner::MemcpyAsyncTaskInfo;
using MemcpyAsyncTaskInfoPtr = std::shared_ptr<MemcpyAsyncTaskInfo>;

namespace mindspore {
namespace kernel {
AssignKernel::AssignKernel() {}

AssignKernel::~AssignKernel() {}

bool AssignKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
                          const std::vector<AddressPtr> & /*outputs*/, void *stream_ptr) {
  if (inputs.size() != 2) {
    MS_LOG(ERROR) << "inputs size is not two";
    return false;
  }

  if (inputs[0]->addr == inputs[1]->addr) {
    MS_LOG(INFO) << "first addr is same with second addr , no need assign";
    return true;
  }
  rtError_t status = rtMemcpyAsync(inputs[0]->addr, inputs[0]->size, inputs[1]->addr, inputs[1]->size,
                                   RT_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Assign op rtMemcpyAsync failed!";
    return false;
  }
  return true;
}

std::vector<TaskInfoPtr> AssignKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  if (inputs.size() != 2) {
    MS_LOG(EXCEPTION) << "inputs size is not two";
  }
  stream_id_ = stream_id;

  std::shared_ptr<MemcpyAsyncTaskInfo> task_info_ptr = std::make_shared<MemcpyAsyncTaskInfo>(
    stream_id, inputs[0]->addr, inputs[0]->size, inputs[1]->addr, inputs[1]->size, RT_MEMCPY_DEVICE_TO_DEVICE);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
