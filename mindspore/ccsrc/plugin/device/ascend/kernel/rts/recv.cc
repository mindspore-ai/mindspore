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

#include "plugin/device/ascend/kernel/rts/recv.h"
#include "runtime/stream.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
using mindspore::ge::model_runner::EventWaitTaskInfo;
using EventWaitTaskInfoPtr = std::shared_ptr<EventWaitTaskInfo>;

RecvKernel::~RecvKernel() {}

bool RecvKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (!common::AnfAlgo::HasNodeAttr(kAttrEventId, anf_node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "RecvKernel has no attr kAttrEventId";
  }
  event_id_ = GetValue<uint32_t>(primitive->GetAttr(kAttrEventId));

  if (common::AnfAlgo::HasNodeAttr(kAttrWaitEvent, anf_node->cast<CNodePtr>())) {
    event_ = reinterpret_cast<rtEvent_t>(GetValue<uintptr_t>(primitive->GetAttr(kAttrWaitEvent)));
  }
  MS_LOG(INFO) << "recv op event_id_:" << event_id_;
  return true;
}

bool RecvKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                        const std::vector<AddressPtr> &, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto status = rtStreamWaitEvent(stream_ptr, event_);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Recv rtStreamWaitEvent failed!";
    return false;
  }

  status = rtEventReset(event_, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtEventReset failed, ret:" << status;
  }
  return true;
}

std::vector<TaskInfoPtr> RecvKernel::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &, uint32_t stream_id) {
  MS_LOG(INFO) << "RecvKernel GenTask event_id_:" << event_id_ << ", stream_id_:" << stream_id;
  stream_id_ = stream_id;
  EventWaitTaskInfoPtr task_info_ptr = std::make_shared<EventWaitTaskInfo>(unique_name_, stream_id, event_id_);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
