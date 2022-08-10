/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ge_runtime/task/end_graph_task.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "runtime/kernel.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
EndGraphTask::EndGraphTask(const ModelContext &model_context, const std::shared_ptr<EndGraphTaskInfo> &task_info)
    : TaskRepeater<EndGraphTaskInfo>(model_context, task_info), task_info_(task_info), stream_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info);
  auto stream_list = model_context.stream_list();
  uint32_t stream_id = task_info->stream_id();
  if (stream_id >= stream_list.size()) {
    MS_LOG(EXCEPTION) << "Stream_id " << stream_id << " is larger than stream_list size " << stream_list.size();
  }
  stream_ = stream_list[stream_id];
  rt_model_handle_ = model_context.rt_model_handle();
}

EndGraphTask::~EndGraphTask() {}

void EndGraphTask::Distribute() {
  MS_LOG(INFO) << "EndGraphTask Distribute start.";
  auto dump_flag = task_info_->dump_flag() ? RT_KERNEL_DUMPFLAG : RT_KERNEL_DEFAULT;
  rtError_t rt_ret = rtEndGraphEx(rt_model_handle_, stream_, dump_flag);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtEndGraphEx failed, ret: " << rt_ret;
  }
  MS_LOG(INFO) << "DistributeTask end";
}

REGISTER_TASK(TaskInfoType::END_GRAPH, EndGraphTask, EndGraphTaskInfo);
}  // namespace mindspore::ge::model_runner
