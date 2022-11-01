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

#include "plugin/device/ascend/hal/device/ge_runtime/task/stream_switch_task.h"
#include "runtime/kernel.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"
#include "mindspore/core/utils/convert_utils_base.h"

namespace mindspore::ge::model_runner {
StreamSwitchTask::StreamSwitchTask(const ModelContext &model_context,
                                   const std::shared_ptr<StreamSwitchTaskInfo> &task_info)
    : TaskRepeater<StreamSwitchTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      stream_list_() {
  MS_EXCEPTION_IF_NULL(task_info);
  stream_list_ = model_context.stream_list();
  if (stream_list_.size() == 1) {
    stream_ = stream_list_[0];
  } else if (stream_list_.size() > task_info->stream_id()) {
    stream_ = stream_list_[task_info->stream_id()];
  } else {
    MS_LOG(EXCEPTION) << "Index: " << task_info->stream_id() << " >= stream_list.size(): " << stream_list_.size();
  }
}

StreamSwitchTask::~StreamSwitchTask() {}

void StreamSwitchTask::Distribute() {
  MS_LOG(INFO) << "Init StreamSwitchTask start.";
  MS_EXCEPTION_IF_NULL(task_info_);
  MS_LOG(INFO) << "Stream " << task_info_->stream_id() << " active " << task_info_->true_stream_id();
  MS_EXCEPTION_IF_NULL(stream_);

  if (static_cast<uint64_t>(task_info_->true_stream_id()) >= stream_list_.size()) {
    MS_LOG(EXCEPTION) << "true_stream_id " << task_info_->true_stream_id() << " must be less than stream_list_ size "
                      << stream_list_.size();
  }

  void *input = static_cast<void *>(task_info_->input_addr());
  rtCondition_t cond = static_cast<rtCondition_t>(task_info_->cond());
  void *value = static_cast<void *>(task_info_->value_addr());
  rtStream_t true_stream = stream_list_[LongToSize(task_info_->true_stream_id())];
  rtSwitchDataType_t data_type = static_cast<rtSwitchDataType_t>(task_info_->data_type());

  MS_LOG(INFO) << "InitStreamSwitchTask, cond: " << cond << ", trueStream: " << true_stream
               << ", trueStreamID: " << task_info_->true_stream_id() << ", datatype: " << task_info_->data_type();

  MS_LOG(INFO) << "StreamSwitchTask Distribute Start.";
  rtError_t rt_ret = rtStreamSwitchEx(input, cond, value, true_stream, stream_, data_type);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtStreamSwitchEx failed, ret: " << rt_ret;
  }

  MS_LOG(INFO) << "Distribute StreamSwitch success";
}

REGISTER_TASK(TaskInfoType::STREAM_SWITCH, StreamSwitchTask, StreamSwitchTaskInfo);
}  // namespace mindspore::ge::model_runner
