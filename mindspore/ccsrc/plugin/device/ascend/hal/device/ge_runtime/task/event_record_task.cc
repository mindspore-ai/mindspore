/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ge_runtime/task/event_record_task.h"
#include "runtime/kernel.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
EventRecordTask::EventRecordTask(const ModelContext &model_context,
                                 const std::shared_ptr<EventRecordTaskInfo> &task_info)
    : TaskRepeater<EventRecordTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      event_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info_);
  auto stream_list = model_context.stream_list();
  auto event_list = model_context.event_list();
  uint32_t stream_id = task_info_->stream_id();
  uint32_t event_id = task_info_->event_id();
  if (stream_id >= stream_list.size() || event_id >= event_list.size()) {
    MS_LOG(EXCEPTION) << "stream_list size: " << stream_list.size() << ", stream_id: " << stream_id
                      << ", event_list size: " << event_list.size() << ", event_id: " << event_id;
  }
  stream_ = stream_list[stream_id];
  event_ = event_list[event_id];
}

EventRecordTask::~EventRecordTask() {
  stream_ = nullptr;
  event_ = nullptr;
}

void EventRecordTask::Distribute() {
  MS_EXCEPTION_IF_NULL(task_info_);
  MS_LOG(INFO) << "EventRecordTask Distribute start, stream: " << stream_ << ", event: " << event_
               << ", stream_id: " << task_info_->stream_id() << ", event_id: " << task_info_->event_id();
  rtError_t rt_ret = rtEventRecord(event_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtEventRecord failed, ret: " << rt_ret;
  }
  MS_LOG(INFO) << "Distribute end.";
}

REGISTER_TASK(TaskInfoType::EVENT_RECORD, EventRecordTask, EventRecordTaskInfo);
}  // namespace mindspore::ge::model_runner
