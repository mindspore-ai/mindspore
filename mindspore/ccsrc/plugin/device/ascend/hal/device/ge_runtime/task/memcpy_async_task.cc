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

#include "plugin/device/ascend/hal/device/ge_runtime/task/memcpy_async_task.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
MemcpyAsyncTask::MemcpyAsyncTask(const ModelContext &model_context,
                                 const std::shared_ptr<MemcpyAsyncTaskInfo> &task_info)
    : TaskRepeater<MemcpyAsyncTaskInfo>(model_context, task_info), task_info_(task_info), stream_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info);
  auto stream_list = model_context.stream_list();
  uint32_t stream_id = task_info->stream_id();

  MS_LOG(INFO) << "Stream list size: " << stream_list.size() << ", stream id: " << stream_id;
  if (stream_id >= stream_list.size()) {
    MS_LOG(EXCEPTION) << "Index: " << task_info->stream_id() << " >= stream_list.size(): " << stream_list.size();
  }
  stream_ = stream_list[stream_id];
}

MemcpyAsyncTask::~MemcpyAsyncTask() {}

void MemcpyAsyncTask::Distribute() {
  MS_EXCEPTION_IF_NULL(task_info_);
  MS_LOG(INFO) << "MemcpyAsyncTask Distribute start.";
  MS_LOG(INFO) << "dst_max: " << task_info_->dst_max() << ", count: " << task_info_->count()
               << ", kind: " << task_info_->kind();
  rtError_t rt_ret = aclrtMemcpyAsync(task_info_->dst(), task_info_->dst_max(), task_info_->src(), task_info_->count(),
                                      static_cast<aclrtMemcpyKind>(task_info_->kind()), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api aclrtMemcpyAsync failed, ret: " << rt_ret;
  }
  MS_LOG(INFO) << "DistributeTask end";
}

REGISTER_TASK(TaskInfoType::MEMCPY_ASYNC, MemcpyAsyncTask, MemcpyAsyncTaskInfo);
}  // namespace mindspore::ge::model_runner
