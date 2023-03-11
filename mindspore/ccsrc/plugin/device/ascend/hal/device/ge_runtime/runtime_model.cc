/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ge_runtime/runtime_model.h"
#include <set>
#include "runtime/kernel.h"
#include "runtime/rt_model.h"
#include "external/runtime/rt_error_codes.h"
#include "plugin/device/ascend/hal/device/ge_runtime/model_context.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"
#include "mindspore/core/utils/log_adapter.h"
#include "include/common/utils/utils.h"
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#endif

namespace mindspore::ge::model_runner {
RuntimeModel::~RuntimeModel() {
  MS_LOG(INFO) << "RuntimeModel destructor start.";

  // Unbind rtModel from all task related streams
  RtModelUnbindStream();

  // Release task first, hccl task hold stream
  task_list_.clear();

  // Release all task related streams
  RtStreamDestory();

  // Release rtlabel resource
  RtLabelDestory();

  // Release rtEvent resourece
  RtEventDestory();

  MS_LOG(INFO) << "Do RtModelDestroy";
  // Release all rt_model
  RtModelDestory();
}

void RuntimeModel::InitStream(const std::shared_ptr<DavinciModel> &davinci_model) {
  MS_EXCEPTION_IF_NULL(davinci_model);

  std::set<int64_t> wait_active_streams;
  std::set<int64_t> force_copy_streams;

  for (const auto &stream_id : davinci_model->GetWaitActiveStreams()) {
    MS_LOG(INFO) << "Stream id " << stream_id << " is wait active stream.";
    (void)wait_active_streams.insert(stream_id);
  }

  for (const auto &stream_id : davinci_model->GetForceCopyStreams()) {
    MS_LOG(INFO) << "Stream id " << stream_id << " is force copy stream.";
    (void)force_copy_streams.insert(stream_id);
  }

  MS_LOG(INFO) << "Total stream num " << davinci_model->GetStreamNum();
  for (uint32_t i = 0; i < davinci_model->GetStreamNum(); ++i) {
    rtStream_t stream = nullptr;
    uint32_t flag = (force_copy_streams.find(i) != force_copy_streams.end())
                      ? (RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY)
                      : (RT_STREAM_PERSISTENT);

    rtError_t rt_ret = rtStreamCreateWithFlags(&stream, davinci_model->GetPriority(), flag);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtStreamCreate failed, ret: " << rt_ret;
    }

    MS_LOG(INFO) << "rtStreamCreateWithFlags end.";
    (void)stream_list_.emplace_back(stream);

    // Bind rt_model_handle_ to all task related streams
    flag = (wait_active_streams.find(i) != wait_active_streams.end()) ? (static_cast<uint32_t>(RT_INVALID_FLAG))
                                                                      : (static_cast<uint32_t>(RT_HEAD_STREAM));
    rt_ret = rtModelBindStream(rt_model_handle_, stream, flag);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtModelBindStream failed, ret: " << rt_ret;
    }
    MS_LOG(INFO) << "stream index: " << i << ", stream: " << stream;
  }
}

void RuntimeModel::InitEvent(uint32_t event_num) {
  MS_LOG(INFO) << "Event number: " << event_num;
  for (uint32_t i = 0; i < event_num; ++i) {
    rtEvent_t rt_event;
    rtError_t rt_ret = rtEventCreateWithFlag(&rt_event, RT_EVENT_WITH_FLAG);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtEventCreate failed, ret: " << rt_ret;
    }
    event_list_.push_back(rt_event);
  }
}

void RuntimeModel::InitLabel(const std::shared_ptr<DavinciModel> &davinci_model) {
  MS_LOG(INFO) << "Label number: " << davinci_model->GetBatchNum();
  label_list_.resize(davinci_model->GetBatchNum());
  for (auto &task_info : davinci_model->GetTaskInfoList()) {
    MS_EXCEPTION_IF_NULL(task_info);

    if (task_info->type() != TaskInfoType::LABEL_SET) {
      continue;
    }
    auto label_set_task_info = std::static_pointer_cast<LabelSetTaskInfo>(task_info);
    MS_EXCEPTION_IF_NULL(label_set_task_info);
    if (label_set_task_info->stream_id() >= stream_list_.size()) {
      MS_LOG(EXCEPTION) << "Invalid stream id " << label_set_task_info->stream_id() << " total stream num "
                        << stream_list_.size();
    }

    rtLabel_t rt_label = nullptr;
    rtError_t rt_ret = rtLabelCreateExV2(&rt_label, rt_model_handle_, stream_list_[label_set_task_info->stream_id()]);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtLabelCreate failed, ret: " << rt_ret
                        << "\nIf you have set MS_COMM_COMPILER_OPT, notice that it will increase labels used and "
                        << "may exceed the maximum label number: 1024. For more details, please refer to"
                        << " 'MS_COMM_COMPILER_OPT' at https://www.mindspore.cn .";
    }
    label_list_[label_set_task_info->label_id()] = rt_label;
  }
}

void RuntimeModel::InitResource(const std::shared_ptr<DavinciModel> &davinci_model) {
  MS_LOG(INFO) << "InitResource start";
  MS_EXCEPTION_IF_NULL(davinci_model);

  rtError_t rt_ret = rtModelCreate(&rt_model_handle_, 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtModelCreate failed, ret: " << rt_ret;
  }

  rt_model_stream_ = davinci_model->model_stream();

  InitStream(davinci_model);
  InitEvent(davinci_model->GetEventNum());
  InitLabel(davinci_model);

  MS_LOG(INFO) << "InitResource success";
}

void RuntimeModel::GenerateTask(uint32_t device_id, uint64_t session_id,
                                const std::shared_ptr<DavinciModel> &davinci_model) {
  MS_LOG(INFO) << "GenerateTask start.";
  MS_EXCEPTION_IF_NULL(davinci_model);
  auto task_infos = davinci_model->GetTaskInfoList();
  ModelContext model_context(device_id, session_id, davinci_model->GetPriority(), rt_model_handle_, rt_model_stream_,
                             stream_list_, label_list_, event_list_);
  for (auto &task_info : task_infos) {
    auto task = TaskFactory::GetInstance().Create(model_context, task_info);
    task_list_.push_back(task);
  }
  MS_LOG(INFO) << "GenerateTask success.";
}

void RuntimeModel::LoadComplete() {
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  auto rt_ret = rtModelGetTaskId(rt_model_handle_, &task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtModelGetTaskId failed, ret: " << rt_ret;
  }
  task_id_list_.push_back(task_id);
  stream_id_list_.push_back(stream_id);

  rt_ret = rtModelLoadComplete(rt_model_handle_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtModelLoadComplete failed, ret: " << rt_ret;
  }
}

void RuntimeModel::Load(uint32_t device_id, uint64_t session_id, const std::shared_ptr<DavinciModel> &davinci_model) {
  InitResource(davinci_model);
  GenerateTask(device_id, session_id, davinci_model);
}

void RuntimeModel::DistributeTask() {
  MS_LOG(INFO) << "DistributeTask start.";
  for (auto &task : task_list_) {
    MS_EXCEPTION_IF_NULL(task);
    task->set_model_handle(rt_model_handle_);
    task->Distribute();

    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    rtError_t rt_ret = rtModelGetTaskId(rt_model_handle_, &task_id, &stream_id);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtModelGetTaskId failed, ret: " << rt_ret;
    }
    task_id_list_.push_back(task_id);
    stream_id_list_.push_back(stream_id);
    if (task->Args() != nullptr) {
      std::shared_ptr<RuntimeInfo> runtime_tuple = std::make_shared<RuntimeInfo>(task_id, stream_id, task->Args());
      auto emplace_ret = runtime_info_map_.emplace(task->task_name(), runtime_tuple);
      if (!emplace_ret.second) {
        // The task_name is (fullname_with_scope + UniqueId). There should be no duplication.
        MS_LOG(EXCEPTION) << "Task name exist: " << task->task_name();
      }
    }
    if (task->task_name() == kEndGraph) {
      (void)end_graph_info_map_.emplace(task_id, stream_id);
    }
  }
  if (task_list_.empty()) {
    MS_LOG(EXCEPTION) << "Task list is empty";
  }

  MS_LOG(INFO) << "DistributeTask success.";
}

void RuntimeModel::Run() const {
  MS_LOG(INFO) << "Davinci task run start.";
  rtError_t ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0);
  if (ret != RT_ERROR_NONE) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG(EXCEPTION) << "Call rt api rtModelLoadComplete failed, ret: " << ret;
  }

  MS_LOG(INFO) << "Run rtModelExecute success, start to rtStreamSynchronize.";
  ret = rtStreamSynchronize(rt_model_stream_);
  if (ret != RT_ERROR_NONE) {
    if (ret == ACL_ERROR_RT_END_OF_SEQUENCE) {
      MS_LOG(INFO) << "Model stream ACL_ERROR_RT_END_OF_SEQUENCE signal received.";
      return;
    }
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG(EXCEPTION) << "Call rt api rtStreamSynchronize failed, ret: " << ret;
  }

  MS_LOG(INFO) << "Davinci task run success.";
}

void RuntimeModel::RtModelUnbindStream() noexcept {
  for (size_t i = 0; i < stream_list_.size(); i++) {
    if (rtModelUnbindStream(rt_model_handle_, stream_list_[i]) != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Unbind stream from model failed! Index: " << i;
      return;
    }
  }
}

void RuntimeModel::RtStreamDestory() noexcept {
  for (size_t i = 0; i < stream_list_.size(); i++) {
    if (rtStreamDestroy(stream_list_[i]) != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy stream failed! Index: " << i;
      return;
    }
  }
}

void RuntimeModel::RtLabelDestory() noexcept {
  for (size_t i = 0; i < label_list_.size(); i++) {
    if (label_list_[i] == nullptr) {
      continue;
    }
    if (rtLabelDestroy(label_list_[i]) != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy label failed! Index: " << i;
      return;
    }
  }
}

void RuntimeModel::RtModelDestory() const noexcept {
  rtError_t ret = rtModelDestroy(rt_model_handle_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtModelDestroy failed, ret: " << ret;
    return;
  }
}

void RuntimeModel::RtEventDestory() noexcept {
  for (size_t i = 0; i < event_list_.size(); i++) {
    if (rtEventDestroy(event_list_[i]) != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy event failed! Index: " << i;
      return;
    }
  }
}

const std::vector<uint32_t> &RuntimeModel::GetTaskIdList() const { return task_id_list_; }

const std::vector<std::shared_ptr<Task>> &RuntimeModel::GetTaskList() const { return task_list_; }

const std::vector<uint32_t> &RuntimeModel::GetStreamIdList() const { return stream_id_list_; }
}  // namespace mindspore::ge::model_runner
