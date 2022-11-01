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

#include "plugin/device/ascend/hal/device/ge_runtime/task/profiler_task.h"
#include <limits>
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
ProfilerTask::ProfilerTask(const ModelContext &model_context, const std::shared_ptr<ProfilerTraceTaskInfo> &task_info)
    : TaskRepeater<ProfilerTraceTaskInfo>(model_context, task_info), task_info_(task_info), stream_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info);
  auto stream_list = model_context.stream_list();
  uint32_t stream_id = task_info->stream_id();
  MS_LOG(INFO) << "Stream list size: " << stream_list.size() << ", stream id: " << stream_id;
  if (stream_id >= stream_list.size()) {
    MS_LOG(EXCEPTION) << "Index: " << task_info->stream_id() << " >= stream_list.size(): " << stream_list.size();
  }
  stream_ = stream_list[stream_id];
}

ProfilerTask::~ProfilerTask() { stream_ = nullptr; }

void ProfilerTask::Distribute() {
  MS_LOG(INFO) << "ProfilerTask Distribute start.";
  MS_EXCEPTION_IF_NULL(task_info_);
  uint32_t model_id = 0;
  rtError_t rt_model_ret = rtModelGetId(model_handle_, &model_id);
  if (rt_model_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtModelGetId failed, ret: " << rt_model_ret;
  }
  auto new_model_id = static_cast<uint64_t>(model_id);
  // The first step index, here keep same with ge
  uint64_t first_step_index = 1;
  if (task_info_->log_id() > static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
    MS_LOG(EXCEPTION) << "Invalid log id " << task_info_->log_id() << " over max uint16_t.";
  }
  rtError_t rt_ret =
    rtProfilerTraceEx(first_step_index, new_model_id, static_cast<uint16_t>(task_info_->log_id()), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtProfilerTraceEx failed, ret: " << rt_ret;
  }

  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  rtError_t task_ret = rtModelGetTaskId(model_handle_, &task_id, &stream_id);
  if (task_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtModelGetTaskId failed, ret: " << task_ret;
  }

  MS_LOG(INFO) << "[Profiling] Report step point, model id:" << new_model_id << ", stream id: " << stream_id
               << ", task id: " << task_id << ", tag: " << task_info_->log_id();

  MS_LOG(INFO) << "DistributeTask end.";
}

REGISTER_TASK(TaskInfoType::PROFILER_TRACE, ProfilerTask, ProfilerTraceTaskInfo);
}  // namespace mindspore::ge::model_runner
