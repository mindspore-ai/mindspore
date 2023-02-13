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

#include "plugin/device/ascend/hal/device/ge_runtime/task/label_goto_task.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"

namespace mindspore::ge::model_runner {
LabelGotoTask::LabelGotoTask(const ModelContext &model_context, const std::shared_ptr<LabelGotoTaskInfo> &task_info)
    : TaskRepeater<LabelGotoTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      index_value_(nullptr),
      label_id_(task_info_->label_id()),
      rt_model_handle_(model_context.rt_model_handle()) {
  MS_EXCEPTION_IF_NULL(task_info_);
  auto stream_list = model_context.stream_list();
  auto label_list = model_context.label_list();
  uint32_t stream_id = task_info_->stream_id();
  MS_LOG(INFO) << "Stream list size: " << stream_list.size() << ", stream id: " << stream_id;
  MS_LOG(INFO) << "Label list size: " << label_list.size() << ", label id: " << label_id_;
  if (stream_id >= stream_list.size() || label_id_ >= label_list.size()) {
    MS_LOG(EXCEPTION) << "Stream/Label id invalid.";
  }
  stream_ = stream_list[stream_id];
  label_manager_ = LabelManager::GetInstance();
  MS_EXCEPTION_IF_NULL(label_manager_);
  label_info_ = label_manager_->GetLabelInfo(rt_model_handle_, {label_id_}, label_list);
  MS_EXCEPTION_IF_NULL(label_info_);
}

LabelGotoTask::~LabelGotoTask() {
  if (index_value_ != nullptr) {
    rtError_t rt_ret = rtFree(index_value_);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rtFree index_value_ failed, ret: " << rt_ret;
    }
    index_value_ = nullptr;
  }
  stream_ = nullptr;
  rt_model_handle_ = nullptr;
}

void LabelGotoTask::Distribute() {
  MS_LOG(INFO) << "LabelGotoTask Distribute start.";
  MS_EXCEPTION_IF_NULL(stream_);
  MS_EXCEPTION_IF_NULL(label_info_);

  if (index_value_ == nullptr) {
    rtError_t rt_ret = rtMalloc(&index_value_, sizeof(uint64_t), RT_MEMORY_HBM, 0);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtMalloc failed, ret: " << rt_ret;
    }

    uint64_t index = 0;
    rt_ret = aclrtMemcpy(index_value_, sizeof(uint64_t), &index, sizeof(index), ACL_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtMemcpy failed, ret: " << rt_ret;
    }
  }

  void *label_info = label_info_->GetLabelInfo();
  rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, 1, label_info, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtLabelSwitchByIndex failed, ret: " << rt_ret;
  }

  MS_LOG(INFO) << "DistributeTask end.";
}

REGISTER_TASK(TaskInfoType::LABEL_GOTO, LabelGotoTask, LabelGotoTaskInfo);
}  // namespace mindspore::ge::model_runner
