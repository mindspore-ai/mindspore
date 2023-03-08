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

#include "plugin/device/ascend/kernel/rts/label_set.h"
#include "runtime/stream.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

using mindspore::ge::model_runner::LabelSetTaskInfo;
using LabelSetTaskInfoPtr = std::shared_ptr<LabelSetTaskInfo>;

namespace mindspore {
namespace kernel {
LabelSetKernel::~LabelSetKernel() {}

bool LabelSetKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "LabelSetKernel init";
  auto cnode = anf_node->cast<CNodePtr>();
  if (!common::AnfAlgo::HasNodeAttr(kAttrLabelIndex, cnode)) {
    MS_LOG(EXCEPTION) << "LabelSetKernel has no attr label_index";
  }
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  label_ = GetValue<uint32_t>(primitive->GetAttr(kAttrLabelIndex));
  MS_LOG(INFO) << "LabelSetKernel get attr label:" << label_;
  return true;
}

bool LabelSetKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                            const std::vector<AddressPtr> &, void *) {
  MS_LOG(INFO) << "LabelSetKernel launch";
  return true;
}

std::vector<TaskInfoPtr> LabelSetKernel::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &, uint32_t stream_id) {
  MS_LOG(INFO) << "LabelSetKernel GenTask label:" << label_ << ", stream id:" << stream_id;
  std::vector<TaskInfoPtr> task_info_list;
  std::shared_ptr<LabelSetTaskInfo> task_info_ptr = std::make_shared<LabelSetTaskInfo>(unique_name_, stream_id, label_);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  (void)task_info_list.emplace_back(task_info_ptr);
  return task_info_list;
}
}  // namespace kernel
}  // namespace mindspore
