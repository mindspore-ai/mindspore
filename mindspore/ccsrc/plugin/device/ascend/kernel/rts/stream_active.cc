/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/rts/stream_active.h"
#include "runtime/stream.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

using mindspore::ge::model_runner::StreamActiveTaskInfo;
using StreamActiveTaskInfoPtr = std::shared_ptr<StreamActiveTaskInfo>;

namespace mindspore {
namespace kernel {
StreamActiveKernel::StreamActiveKernel() { active_streams_index_ = {}; }

StreamActiveKernel::~StreamActiveKernel() {}

bool StreamActiveKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "stream active op init start";
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (!common::AnfAlgo::HasNodeAttr(kAttrActiveStreamList, anf_node->cast<CNodePtr>())) {
    MS_LOG(EXCEPTION) << "StreamActiveKernel " << anf_node->DebugString() << "has no attr kAttrActiveStreamList";
  }
  active_streams_index_ = GetValue<std::vector<uint32_t>>(primitive->GetAttr(kAttrActiveStreamList));
  return true;
}

bool StreamActiveKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &, void *stream_ptr) {
  MS_LOG(INFO) << "Stream active op launch start";

  if (active_streams_index_.empty()) {
    MS_LOG(ERROR) << "activeStreamList_ is empty!";
    return false;
  }

  rtStream_t act_stream;
  rtError_t status;
  MS_EXCEPTION_IF_NULL(kernel::TaskStream::GetInstance());
  auto stream_list = kernel::TaskStream::GetInstance()->gen_stream_list();
  for (auto index : active_streams_index_) {
    if (index >= stream_list.size()) {
      MS_LOG(EXCEPTION) << "Invalid index: " << index << " stream_list size: " << stream_list.size();
    }
    act_stream = stream_list[index];
    status = rtStreamActive(act_stream, stream_ptr);
    if (status != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Stream active failed!";
      return false;
    }
  }
  return true;
}

std::vector<TaskInfoPtr> StreamActiveKernel::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                     const std::vector<AddressPtr> &, uint32_t stream_id) {
  MS_LOG(INFO) << "StreamActiveKernel GenTask active stream size:" << active_streams_index_.size()
               << ", stream id:" << stream_id;
  stream_id_ = stream_id;
  std::vector<TaskInfoPtr> task_info_list;
  for (auto &index : active_streams_index_) {
    std::shared_ptr<StreamActiveTaskInfo> task_info_ptr =
      std::make_shared<StreamActiveTaskInfo>(unique_name_, stream_id, index);
    MS_EXCEPTION_IF_NULL(task_info_ptr);
    (void)task_info_list.emplace_back(task_info_ptr);
    MS_LOG(INFO) << "StreamActiveKernel GenTask: streamId:" << stream_id << ", Active streamId:" << index;
  }
  return task_info_list;
}
}  // namespace kernel
}  // namespace mindspore
