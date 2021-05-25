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

#include "runtime/device/ascend/ge_runtime/task/label_set_task.h"
#include "runtime/device/ascend/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
LabelSetTask::LabelSetTask(const ModelContext &model_context, const std::shared_ptr<LabelSetTaskInfo> &task_info)
    : TaskRepeater<LabelSetTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      label_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info_);
  auto stream_list = model_context.stream_list();
  auto label_list = model_context.label_list();
  uint32_t stream_id = task_info->stream_id();
  uint32_t label_id = task_info->label_id();
  MS_LOG(INFO) << "Stream list size: " << stream_list.size() << ", stream id: " << stream_id;
  MS_LOG(INFO) << "Label list size: " << label_list.size() << ", label id: " << label_id;
  if (stream_id >= stream_list.size() || label_id >= label_list.size()) {
    MS_LOG(EXCEPTION) << "Stream/Label id invalid.";
  }
  stream_ = stream_list[stream_id];
  label_ = label_list[label_id];
}

LabelSetTask::~LabelSetTask() {}

void LabelSetTask::Distribute() {
  MS_LOG(INFO) << "LabelSetTask Distribute start.";
  MS_EXCEPTION_IF_NULL(stream_);
  MS_EXCEPTION_IF_NULL(label_);

  rtError_t rt_ret = rtLabelSet(label_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtLabelSet failed, ret: " << rt_ret;
  }

  MS_LOG(INFO) << "DistributeTask end.";
}

REGISTER_TASK(TaskInfoType::LABEL_SET, LabelSetTask, LabelSetTaskInfo);
}  // namespace mindspore::ge::model_runner
