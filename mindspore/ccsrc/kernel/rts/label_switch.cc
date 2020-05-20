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

#include "kernel/rts/label_switch.h"
#include <asm-generic/param.h>
#include <memory>
#include "runtime/stream.h"
#include "framework/ge_runtime/task_info.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"

using ge::model_runner::LabelSwitchTaskInfo;
using LabelSwitchTaskInfoPtr = std::shared_ptr<LabelSwitchTaskInfo>;

namespace mindspore {
namespace kernel {
LabelSwitchKernel::LabelSwitchKernel() {
  label_list_ = {};
  cond_ = nullptr;
  label_size_ = 0;
}

LabelSwitchKernel::~LabelSwitchKernel() {}

bool LabelSwitchKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "LabelSwitchKernel init";
  auto cnode = anf_node->cast<CNodePtr>();
  if (!AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, cnode)) {
    MS_LOG(EXCEPTION) << "LabelSwitchKernel has no attr label_switch_list";
  }
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  label_list_ = GetValue<std::vector<uint32_t>>(primitive->GetAttr(kAttrLabelSwitchList));
  label_size_ = label_list_.size();
  MS_LOG(INFO) << "LabelSwitchKernel get attr label size:" << label_size_;
  for (auto label : label_list_) {
    MS_LOG(INFO) << "label: " << label;
  }
  return true;
}

bool LabelSwitchKernel::Launch(const std::vector<AddressPtr> & /*inputs*/,
                               const std::vector<AddressPtr> & /*workspace*/,
                               const std::vector<AddressPtr> & /*outputs*/, void * /*stream_ptr*/) {
  MS_LOG(INFO) << "LabelSwitchKernel launch";
  return true;
}

std::vector<TaskInfoPtr> LabelSwitchKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &workspace,
                                                    const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  MS_LOG(INFO) << "LabelSwitchKernel GenTask label size:" << label_size_ << ", stream id:" << stream_id;
  std::vector<TaskInfoPtr> task_info_list;
  cond_ = inputs[0]->addr;
  //  std::shared_ptr<LabelSwitchTaskInfo> task_info_ptr =
  //    std::make_shared<LabelSwitchTaskInfo>(stream_id, label_size_, &label_list_, cond_);
  // need updata ge task info define
  std::shared_ptr<LabelSwitchTaskInfo> task_info_ptr = std::make_shared<LabelSwitchTaskInfo>(stream_id, label_size_);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  task_info_list.emplace_back(task_info_ptr);
  return task_info_list;
}
}  // namespace kernel
}  // namespace mindspore
