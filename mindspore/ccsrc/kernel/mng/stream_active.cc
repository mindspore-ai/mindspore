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

#include "kernel/mng/stream_active.h"
#include <asm-generic/param.h>
#include <memory>
#include "runtime/stream.h"
#include "framework/ge_runtime/task_info.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"

using ge::model_runner::StreamActiveTaskInfo;
using StreamActiveTaskInfoPtr = std::shared_ptr<StreamActiveTaskInfo>;

namespace mindspore {
namespace kernel {
StreamActiveKernel::StreamActiveKernel() { active_streams_index_ = {}; }

StreamActiveKernel::~StreamActiveKernel() {}

bool StreamActiveKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "stream active op init start";
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  active_streams_index_ = GetValue<std::vector<uint32_t>>(primitive->GetAttr(kAttrActiveStreamList));
  return true;
}

bool StreamActiveKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
  MS_LOG(INFO) << "Stream active op launch start";
  auto stream = reinterpret_cast<rtStream_t>(stream_ptr);

  if (active_streams_index_.empty()) {
    MS_LOG(ERROR) << "activeStreamList_ is empty!";
    return false;
  }

  rtStream_t act_stream;
  rtError_t status;
  for (auto index : active_streams_index_) {
    act_stream = kernel::TaskStream::GetInstance()->gen_stream_list()[index];
    status = rtStreamActive(act_stream, stream);
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
  std::vector<TaskInfoPtr> task_info_list;
  for (auto &index : active_streams_index_) {
    std::shared_ptr<StreamActiveTaskInfo> task_info_ptr = std::make_shared<StreamActiveTaskInfo>(stream_id, index);
    MS_EXCEPTION_IF_NULL(task_info_ptr);
    task_info_list.emplace_back(task_info_ptr);
    MS_LOG(INFO) << "StreamActiveKernel GenTask: streamId:" << stream_id << ", Active streamId:" << index;
  }
  return task_info_list;
}
}  // namespace kernel
}  // namespace mindspore
