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

#include "kernel/mng/recv.h"
#include <memory>
#include "runtime/stream.h"
#include "utils/context/ms_context.h"
#include "device/ascend/ascend_stream_assign.h"
#include "framework/ge_runtime/task_info.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"

namespace mindspore {
namespace kernel {
using ge::model_runner::EventWaitTaskInfo;
using mindspore::device::ascend::AscendStreamAssign;
using EventWaitTaskInfoPtr = std::shared_ptr<EventWaitTaskInfo>;

RecvKernel::RecvKernel() { event_id_ = 0; }

RecvKernel::~RecvKernel() {}

bool RecvKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  event_id_ = GetValue<uint32_t>(primitive->GetAttr(kAttrEventId));
  MS_LOG(INFO) << "recv op event_id_:" << event_id_;
  return true;
}

bool RecvKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                        const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
  rtEvent_t stream_event{};
  auto stream = reinterpret_cast<rtStream_t>(stream_ptr);
  auto status = rtStreamWaitEvent(stream, stream_event);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Recv rtStreamWaitEvent failed!";
    return false;
  }
  return true;
}

std::vector<TaskInfoPtr> RecvKernel::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &, uint32_t stream_id) {
  MS_LOG(INFO) << "RecvKernel GenTask event_id_:" << event_id_ << ", stream_id_:" << stream_id;
  stream_id_ = stream_id;
  EventWaitTaskInfoPtr task_info_ptr = std::make_shared<EventWaitTaskInfo>(stream_id, event_id_);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
