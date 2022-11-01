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

#include "plugin/device/ascend/hal/device/ge_runtime/task/stream_active_task.h"
#include "runtime/kernel.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
StreamActiveTask::StreamActiveTask(const ModelContext &model_context,
                                   const std::shared_ptr<StreamActiveTaskInfo> &task_info)
    : TaskRepeater<StreamActiveTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      active_stream_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info);
  auto stream_list = model_context.stream_list();
  uint32_t stream_id = task_info->stream_id();
  uint32_t active_stream_id = task_info->active_stream_id();
  MS_LOG(INFO) << "Stream list size: " << stream_list.size() << ", stream id: " << stream_id
               << ", active stream id: " << active_stream_id;
  if (stream_id >= stream_list.size() || active_stream_id >= stream_list.size()) {
    MS_LOG(EXCEPTION) << "Stream id invalid";
  }
  stream_ = stream_list[stream_id];
  active_stream_ = stream_list[active_stream_id];
}

StreamActiveTask::~StreamActiveTask() {}

void StreamActiveTask::Distribute() {
  MS_LOG(INFO) << "Distribute start";
  MS_EXCEPTION_IF_NULL(task_info_);
  MS_LOG(INFO) << "Stream " << task_info_->stream_id() << " active " << task_info_->active_stream_id();
  MS_EXCEPTION_IF_NULL(stream_);
  MS_EXCEPTION_IF_NULL(active_stream_);
  rtError_t rt_ret = rtStreamActive(active_stream_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtStreamActive failed, ret: " << rt_ret;
  }
  MS_LOG(INFO) << "DistributeTask end.";
}

REGISTER_TASK(TaskInfoType::STREAM_ACTIVE, StreamActiveTask, StreamActiveTaskInfo);
}  // namespace mindspore::ge::model_runner
