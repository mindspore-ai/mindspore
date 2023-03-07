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

#include "plugin/device/ascend/kernel/rts/send.h"
#include "runtime/event.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

using mindspore::ge::model_runner::EventRecordTaskInfo;
using EventRecordTaskInfoPtr = std::shared_ptr<EventRecordTaskInfo>;

namespace mindspore {
namespace kernel {
SendKernel::~SendKernel() {}

bool SendKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (!common::AnfAlgo::HasNodeAttr(kAttrEventId, anf_node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "SendKernel has no attr kAttrEventId";
  }
  event_id_ = GetValue<uint32_t>(primitive->GetAttr(kAttrEventId));

  if (common::AnfAlgo::HasNodeAttr(kAttrRecordEvent, anf_node->cast<CNodePtr>())) {
    event_ = reinterpret_cast<rtEvent_t>(GetValue<uintptr_t>(primitive->GetAttr(kAttrRecordEvent)));
  }
  MS_LOG(INFO) << "send op event id:" << event_id_;
  return true;
}

bool SendKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                        const std::vector<AddressPtr> &, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  rtError_t status = rtEventRecord(event_, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Send op rtEventRecord failed!";
    return false;
  }
  return true;
}

std::vector<TaskInfoPtr> SendKernel::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &, uint32_t stream_id) {
  MS_LOG(INFO) << "SendKernel GenTask event id:" << event_id_ << ", stream id:" << stream_id;
  stream_id_ = stream_id;
  EventRecordTaskInfoPtr task_info_ptr = std::make_shared<EventRecordTaskInfo>(unique_name_, stream_id, event_id_);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
