/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <memory>
#include "device/ascend/profiling/reporter/task_desc_reporter.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/ascend_kernel_mod.h"

namespace mindspore {
namespace device {
namespace ascend {
void TaskDescReporter::ReportData() {
  MS_LOG(INFO) << "cnode_list.size()=" << cnode_list_.size() << " task_ids_.size()=" << task_ids_.size();
  if (cnode_list_.size() != task_ids_.size()) {
    MS_LOG(ERROR) << "cnode list size not equal task ids size";
    return;
  }

  size_t task_index = 0;
  for (const auto &node : cnode_list_) {
    if (AnfAlgo::GetKernelType(node) != TBE_KERNEL && AnfAlgo::GetKernelType(node) != AKG_KERNEL) {
      MS_LOG(WARNING) << "Skip non tbe kernel";
      ++task_index;
      continue;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(ascend_kernel_mod);
    // Check task_id and stream_id valid
    CheckStreamTaskValid(task_index, task_index);
    auto desc_ptr = std::make_shared<TaskDesc>(node->fullname_with_scope(), task_ids_[task_index],
                                               ascend_kernel_mod->block_dim(), stream_ids_[task_index]);
    prof_desc_list_.emplace_back(desc_ptr);
    ++task_index;
  }
  ReportAllLine();
}

void TaskDescReporter::CheckStreamTaskValid(uint32_t task_id, uint32_t stream_id) {
  if (task_id >= task_ids_.size() || stream_id >= stream_ids_.size()) {
    MS_LOG(EXCEPTION) << "Index invalid. task_id:" << task_id << ", task_ids.size:" << task_ids_.size()
                      << ", stream_id:" << stream_id << ", stream_ids.size:" << stream_ids_.size();
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
