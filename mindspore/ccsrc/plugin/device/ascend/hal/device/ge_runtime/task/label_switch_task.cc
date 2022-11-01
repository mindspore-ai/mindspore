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

#include "plugin/device/ascend/hal/device/ge_runtime/task/label_switch_task.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
LabelSwitchTask::LabelSwitchTask(const ModelContext &model_context,
                                 const std::shared_ptr<LabelSwitchTaskInfo> &task_info)
    : TaskRepeater<LabelSwitchTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      rt_model_handle_(model_context.rt_model_handle()),
      label_info_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info);
  auto all_label_resource = model_context.label_list();
  auto stream_list = model_context.stream_list();
  uint32_t stream_id = task_info->stream_id();
  MS_LOG(INFO) << "Stream list size: " << stream_list.size() << ", stream id: " << stream_id;
  if (stream_id >= stream_list.size()) {
    MS_LOG(EXCEPTION) << "Stream id invalid.";
  }
  stream_ = stream_list[stream_id];
  label_manager_ = LabelManager::GetInstance();
  MS_EXCEPTION_IF_NULL(label_manager_);
  label_info_ = label_manager_->GetLabelInfo(rt_model_handle_, task_info_->label_list(), all_label_resource);
  MS_EXCEPTION_IF_NULL(label_info_);
}

LabelSwitchTask::~LabelSwitchTask() {
  stream_ = nullptr;
  rt_model_handle_ = nullptr;
}

void LabelSwitchTask::Distribute() {
  MS_LOG(INFO) << "LabelSwitchTask Distribute start.";
  CheckParamValid();

  void *label_info = label_info_->GetLabelInfo();
  rtError_t rt_ret = rtLabelSwitchByIndex(task_info_->cond(), task_info_->label_size(), label_info, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtLabelSwitchByIndex failed, ret: " << rt_ret;
  }

  MS_LOG(INFO) << "DistributeTask end.";
}

void LabelSwitchTask::CheckParamValid() {
  MS_EXCEPTION_IF_NULL(stream_);
  MS_EXCEPTION_IF_NULL(task_info_);
  if (task_info_->label_list().empty()) {
    MS_LOG(EXCEPTION) << "label_list is empty.";
  }

  if (task_info_->label_size() != task_info_->label_list().size()) {
    MS_LOG(EXCEPTION) << "label_list size " << task_info_->label_list().size() << " but label_size is "
                      << task_info_->label_size();
  }

  if (task_info_->label_size() >= UINT32_MAX / sizeof(rtLabelDevInfo)) {
    MS_LOG(EXCEPTION) << "label_size " << task_info_->label_size() << " will overflow.";
  }
}

REGISTER_TASK(TaskInfoType::LABEL_SWITCH, LabelSwitchTask, LabelSwitchTaskInfo);
}  // namespace mindspore::ge::model_runner
