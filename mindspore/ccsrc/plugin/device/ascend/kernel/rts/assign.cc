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

#include "plugin/device/ascend/kernel/rts/assign.h"
#include <memory>
#include "runtime/mem.h"
#include "acl/acl_rt.h"

using mindspore::ge::model_runner::MemcpyAsyncTaskInfo;
using MemcpyAsyncTaskInfoPtr = std::shared_ptr<MemcpyAsyncTaskInfo>;

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kAssignInputSize = 2;
}
AssignKernel::AssignKernel() {}

AssignKernel::~AssignKernel() {}

bool AssignKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &, void *stream_ptr) {
  if (inputs.size() != kAssignInputSize) {
    MS_LOG(ERROR) << "inputs size is not two";
    return false;
  }

  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  if (inputs[0]->addr == inputs[1]->addr) {
    MS_LOG(INFO) << "first addr is same with second addr , no need assign";
    return true;
  }
  rtError_t status = aclrtMemcpyAsync(inputs[0]->addr, inputs[0]->size, inputs[1]->addr, inputs[1]->size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Assign op aclrtMemcpyAsync failed!";
    return false;
  }
  return true;
}

std::vector<TaskInfoPtr> AssignKernel::GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &, uint32_t stream_id) {
  if (inputs.size() != kAssignInputSize) {
    MS_LOG(EXCEPTION) << "Inputs size should be 2, but got " << inputs.size();
  }
  stream_id_ = stream_id;

  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  std::shared_ptr<MemcpyAsyncTaskInfo> task_info_ptr =
    std::make_shared<MemcpyAsyncTaskInfo>(unique_name_, stream_id, inputs[0]->addr, inputs[0]->size, inputs[1]->addr,
                                          inputs[1]->size, ACL_MEMCPY_DEVICE_TO_DEVICE, false);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
