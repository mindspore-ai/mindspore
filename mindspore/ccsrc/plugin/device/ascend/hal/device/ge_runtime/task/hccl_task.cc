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

#include "plugin/device/ascend/hal/device/ge_runtime/task/hccl_task.h"
#include <algorithm>
#include "plugin/device/ascend/hal/device/ge_runtime/task/task_factory.h"
#include "common/opskernel/ops_kernel_info_store.h"

namespace mindspore::ge::model_runner {
std::map<rtModel_t, std::map<uint32_t, std::vector<std::weak_ptr<HcclTask::StreamGuard>>>>
  HcclTask::model_stream_mapping_;
std::mutex HcclTask::model_stream_mapping_mutex_;

HcclTask::HcclTask(const ModelContext &model_context, const std::shared_ptr<HcclTaskInfo> &task_info)
    : TaskRepeater<HcclTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      workspace_mem_(nullptr),
      rt_model_handle_(nullptr),
      priority_(0),
      secondary_stream_list_() {
  MS_EXCEPTION_IF_NULL(task_info_);

  priority_ = model_context.priority();
  rt_model_handle_ = model_context.rt_model_handle();
  auto stream_list = model_context.stream_list();
  if (stream_list.size() == 1) {
    stream_ = stream_list[0];
  } else if (stream_list.size() > task_info_->stream_id()) {
    stream_ = stream_list[task_info_->stream_id()];
  } else {
    MS_LOG(EXCEPTION) << "Index: " << task_info_->stream_id() << " >= stream_list.size(): " << stream_list.size();
  }
}

HcclTask::~HcclTask() {
  if (task_info_ == nullptr) {
    return;
  }
  ::ge::OpsKernelInfoStore *ops_kernel_info_store =
    static_cast<::ge::OpsKernelInfoStore *>(task_info_->ops_kernel_store());
  if (ops_kernel_info_store != nullptr) {
    (void)ops_kernel_info_store->UnloadTask(ge_task_);
  }
}

void HcclTask::Distribute() {
  // Ops kernel info store
  // Get privateDef and opsKernelStorePtr
  MS_LOG(INFO) << "Distribute hccl task start.";
  MS_EXCEPTION_IF_NULL(task_info_);
  void *ops_kernel_store = task_info_->ops_kernel_store();
  ::ge::OpsKernelInfoStore *ops_kernel_info_store = static_cast<::ge::OpsKernelInfoStore *>(ops_kernel_store);
  MS_EXCEPTION_IF_NULL(ops_kernel_info_store);

  char *private_def = reinterpret_cast<char *>(const_cast<char unsigned *>(task_info_->private_def().data()));
  auto private_def_len = static_cast<uint32_t>(task_info_->private_def().size());
  MS_LOG(INFO) << "The first address of the custom info, privateDef= " << private_def;
  SetSecondaryStream();

  if (task_info_->workspace_size() > 0) {
    workspace_mem_ = task_info_->workspace_addr();
  }

  static uint32_t task_id = 0;
  ge_task_.id = task_id++;
  ge_task_.type = static_cast<uint16_t>(RT_MODEL_TASK_HCCL);
  ge_task_.stream = stream_;

  ge_task_.kernelHcclInfo = std::vector<::ge::GETaskKernelHcclInfo>(1);
  ge_task_.kernelHcclInfo[0].hccl_type = task_info_->hccl_type();
  ge_task_.kernelHcclInfo[0].inputDataAddr = task_info_->input_data_addr();
  ge_task_.kernelHcclInfo[0].outputDataAddr = task_info_->output_data_addr();
  ge_task_.kernelHcclInfo[0].workSpaceAddr = workspace_mem_;
  ge_task_.kernelHcclInfo[0].workSpaceMemSize = static_cast<uint64_t>(task_info_->workspace_size());
  ge_task_.kernelHcclInfo[0].count = task_info_->count();
  ge_task_.kernelHcclInfo[0].dataType = static_cast<int32_t>(task_info_->data_type());
  ge_task_.kernelHcclInfo[0].opType = static_cast<int32_t>(task_info_->op_type());
  ge_task_.kernelHcclInfo[0].rootId = task_info_->root_id();
  if (!task_info_->global_workspace_addr().empty()) {
    ge_task_.kernelHcclInfo[0].global_workspace_addr = task_info_->global_workspace_addr();
  }

  std::vector<rtStream_t> secondary_stream_list;
  std::transform(secondary_stream_list_.begin(), secondary_stream_list_.end(),
                 std::back_inserter(secondary_stream_list),
                 [](const std::shared_ptr<StreamGuard> &stream) -> rtStream_t { return stream->GetStream(); });
  ge_task_.kernelHcclInfo[0].hcclStreamList = secondary_stream_list;

  ge_task_.privateDef = private_def;
  ge_task_.privateDefLen = private_def_len;
  ge_task_.opsKernelStorePtr = ops_kernel_store;

  MS_LOG(INFO) << "Begin to call function LoadTask in hccl. " << task_info_->op_name();
  auto result = ops_kernel_info_store->LoadTask(ge_task_);
  // tagHcclResult::HCCL_SUCCESS is 0
  if (result != 0) {
    MS_LOG(EXCEPTION) << "davinci_model : load task fail, return ret: " << result;
  }

  MS_LOG(INFO) << "Call function LoadTask end.";
}

void HcclTask::SetSecondaryStream() {
  const uint32_t master_stream_id = task_info_->stream_id();
  const int64_t hccl_secondary_stream_num = task_info_->hccl_stream_num();
  std::lock_guard<std::mutex> lock(model_stream_mapping_mutex_);

  // no model, create all secondary stream
  auto model_iter = model_stream_mapping_.find(rt_model_handle_);
  if (model_iter == model_stream_mapping_.end()) {
    MS_LOG(INFO) << "Need to create map for rt_model_handle_: " << rt_model_handle_ << " with new mainstream "
                 << master_stream_id;
    CreateStream(hccl_secondary_stream_num, master_stream_id);
    MS_LOG(INFO) << "Initialize hccl secondary stream success, hccl_secondary_stream_num=" << hccl_secondary_stream_num;
    return;
  }

  // has model, but no secondary stream before, create all secondary stream
  auto &master_secondary_stream_map = model_iter->second;
  auto iter = master_secondary_stream_map.find(master_stream_id);
  if (iter == master_secondary_stream_map.end()) {
    MS_LOG(INFO) << "Need to create secondary stream for " << task_info_->op_name() << " with new mainstream "
                 << master_stream_id;
    CreateStream(hccl_secondary_stream_num, master_stream_id);
    MS_LOG(INFO) << "Initialize hccl secondary stream success, hccl_secondary_stream_num=" << hccl_secondary_stream_num;
    return;
  }

  // has model, has secondary stream, but number is not enough to be reuse
  std::vector<std::weak_ptr<StreamGuard>> &secondary_stream_vec = iter->second;
  if (static_cast<size_t>(hccl_secondary_stream_num) > secondary_stream_vec.size()) {
    size_t created_stream_num = secondary_stream_vec.size();
    auto need_to_create_num = hccl_secondary_stream_num - created_stream_num;
    MS_LOG(INFO) << "Need to reuse " << secondary_stream_vec.size() << " secondary stream and create "
                 << need_to_create_num << " new secondary stream.";
    for (size_t i = 0; i < secondary_stream_vec.size(); ++i) {
      secondary_stream_list_.push_back(GetSecondaryStream(&secondary_stream_vec, i));
    }
    CreateStream(need_to_create_num, master_stream_id);
    MS_LOG(INFO) << "Initialize hccl secondary stream success, hccl_secondary_stream_num=" << hccl_secondary_stream_num;
    return;
  }

  // all can be reuse
  MS_LOG(INFO) << "Number of secondary stream " << hccl_secondary_stream_num << " is enough to be reused.";
  for (int64_t i = 0; i < hccl_secondary_stream_num; ++i) {
    secondary_stream_list_.push_back(GetSecondaryStream(&secondary_stream_vec, i));
  }
  MS_LOG(INFO) << "Initialize hccl secondary stream success, hccl_secondary_stream_num = " << hccl_secondary_stream_num;
}

void HcclTask::CreateStream(int64_t stream_num, int64_t master_stream_id) {
  MS_LOG(INFO) << "Start to create " << stream_num << " hccl secondary stream.";
  for (int64_t i = 0; i < stream_num; ++i) {
    rtStream_t stream = nullptr;
    CreateStream(rt_model_handle_, &stream);
    auto shared_stream = std::make_shared<StreamGuard>(rt_model_handle_, stream);
    SaveHcclSecondaryStream(master_stream_id, shared_stream);
    secondary_stream_list_.push_back(shared_stream);
  }
  MS_LOG(INFO) << "CreateStream success.";
}

void HcclTask::CreateStream(rtModel_t model, rtStream_t *stream) const {
  MS_EXCEPTION_IF_NULL(stream);

  rtError_t rt_ret = rtStreamCreateWithFlags(stream, priority_, RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtEventRecord failed, ret: " << rt_ret;
  }
  // Create secondary stream, inactive by default, activated by hccl
  rt_ret = rtModelBindStream(model, *stream, RT_MODEL_WAIT_ACTIVE_STREAM);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtEventRecord failed, ret: " << rt_ret;
  }
}

void HcclTask::SaveHcclSecondaryStream(int64_t master_stream_id, const std::shared_ptr<StreamGuard> &stream) {
  if (model_stream_mapping_.find(rt_model_handle_) == model_stream_mapping_.end()) {
    model_stream_mapping_.emplace(rt_model_handle_, std::map<uint32_t, std::vector<std::weak_ptr<StreamGuard>>>());
  }
  std::map<uint32_t, std::vector<std::weak_ptr<StreamGuard>>> &master_secondary_stream_map =
    model_stream_mapping_.at(rt_model_handle_);
  master_secondary_stream_map[master_stream_id].emplace_back(stream);
}

HcclTask::StreamGuard::~StreamGuard() {
  rtError_t rt_ret = rtModelUnbindStream(model_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtModelUnbindStream failed, ret: " << rt_ret;
    return;
  }

  rt_ret = rtStreamDestroy(stream_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtStreamDestroy failed, ret: " << rt_ret;
    return;
  }
}

std::shared_ptr<HcclTask::StreamGuard> HcclTask::GetSecondaryStream(
  std::vector<std::weak_ptr<StreamGuard>> *secondary_streams, size_t index) {
  MS_EXCEPTION_IF_NULL(secondary_streams);
  if (index >= secondary_streams->size()) {
    MS_LOG(EXCEPTION) << "Invalid stream index " << index << ", secondary streams size " << secondary_streams->size();
  }
  auto stream = secondary_streams->at(index).lock();
  if (stream == nullptr) {
    rtStream_t new_stream = nullptr;
    CreateStream(rt_model_handle_, &new_stream);
    stream = std::make_shared<HcclTask::StreamGuard>(rt_model_handle_, new_stream);
    (*secondary_streams)[index] = stream;
  }
  return stream;
}

REGISTER_TASK(TaskInfoType::HCCL, HcclTask, HcclTaskInfo);
}  // namespace mindspore::ge::model_runner
