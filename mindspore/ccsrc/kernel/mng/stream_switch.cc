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

#include "kernel/mng/stream_switch.h"

#include <memory>
#include <vector>

#include "runtime/stream.h"
#include "framework/ge_runtime/task_info.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"

using ge::model_runner::StreamSwitchTaskInfo;
using StreamSwitchTaskInfoPtr = std::shared_ptr<StreamSwitchTaskInfo>;

namespace mindspore {
namespace kernel {
StreamSwitchKernel::StreamSwitchKernel() {
  cond_ = RT_EQUAL;
  true_stream_index_ = 0;
  data_type_ = RT_SWITCH_INT32;
}

StreamSwitchKernel::~StreamSwitchKernel() {}

bool StreamSwitchKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "stream switch op init start";
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  cond_ = tagRtCondition(GetValue<int>(primitive->GetAttr(kAttrSwitchCondition)));
  true_stream_index_ = GetValue<uint32_t>(primitive->GetAttr(kAttrTrueBranchStream));
  data_type_ = tagRtSwitchDataType(GetValue<int>(primitive->GetAttr(kAttrDataType)));
  MS_LOG(INFO) << "cond_:" << static_cast<int>(cond_) << ", true_stream_index_:" << true_stream_index_
               << ", data_type_:" << static_cast<int>(data_type_);
  return true;
}

bool StreamSwitchKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
  MS_LOG(INFO) << "stream switch op launch start";
  if (inputs.size() != 2) {
    MS_LOG(ERROR) << "Stream switch inputs size is " << inputs.size() << ", only support 2";
  }

  void *loop_cnt = inputs[0]->addr;
  void *ites_per_loop = inputs[1]->addr;
  auto stream = reinterpret_cast<rtStream_t>(stream_ptr);
  rtStream_t true_stream_ = kernel::TaskStream::GetInstance()->gen_stream_list()[true_stream_index_];
  rtError_t status = rtStreamSwitchEx(loop_cnt, cond_, ites_per_loop, true_stream_, stream, data_type_);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Stream switch failed!";
    return false;
  }
  return true;
}

std::vector<TaskInfoPtr> StreamSwitchKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                     uint32_t stream_id) {
  MS_LOG(INFO) << "StreamSwitchKernel GenTask start";
  if (inputs.size() != 2) {
    MS_LOG(ERROR) << "stream switch inputs size is " << inputs.size() << ", is not two";
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  auto loop_cnt = inputs[0]->addr;
  auto ites_per_loop = inputs[1]->addr;
  MS_LOG(INFO) << "cond_:" << static_cast<int>(cond_) << ", true_stream_index_:" << true_stream_index_
               << ", stream_id:" << stream_id;
  std::shared_ptr<StreamSwitchTaskInfo> task_info_ptr =
    std::make_shared<StreamSwitchTaskInfo>(stream_id, true_stream_index_, loop_cnt, ites_per_loop, cond_, data_type_);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
