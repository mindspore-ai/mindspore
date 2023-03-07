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

#include "plugin/device/ascend/kernel/rts/end_graph.h"
#include <memory>
#include <vector>
#include "runtime/stream.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

using mindspore::ge::model_runner::EndGraphTaskInfo;
using EndGraphTaskInfoPtr = std::shared_ptr<EndGraphTaskInfo>;
namespace {
constexpr size_t kStreamSwitchInputSize = 2;
}
namespace mindspore {
namespace kernel {
EndGraphKernel::~EndGraphKernel() {}

bool EndGraphKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "End graph op init start";
  return true;
}

bool EndGraphKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                            const std::vector<AddressPtr> &, void *stream_ptr) {
  MS_LOG(INFO) << "EndGraphKernel launch";
  return true;
}

std::vector<TaskInfoPtr> EndGraphKernel::GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &, uint32_t stream_id) {
  MS_LOG(INFO) << "EndGraphKernel GenTask start";
  MS_LOG(INFO) << "Stream_id:" << stream_id;
  std::shared_ptr<EndGraphTaskInfo> task_info_ptr =
    std::make_shared<EndGraphTaskInfo>(unique_name_, stream_id, NeedDump());
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
