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
#include "device/ascend/ascend_stream_assign.h"
#include "device/ascend/tasksink/task_generator.h"
#include "device/kernel_adjust.h"

namespace mindspore {
namespace device {
namespace ascend {
void AscendStreamAssign::AssignStreamNew(const KernelGraphPtr &graph) { return; }

uint32_t AscendStreamAssign::GetTotalStreamNum() const { return 1; }

void AscendStreamAssign::GetWaitStreams(vector<uint32_t> *wait_active_stream_list) { return; }

namespace tasksink {
bool TaskGenerator::GenTasks(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *const task_info_list,
                             uint32_t graph_id) {
  return true;
}
}  // namespace tasksink
}  // namespace ascend
void KernelAdjust::Reorder(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) { return; }
void KernelAdjust::InsertSwitchLoop(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) { return; }
bool KernelAdjust::StepLoadCtrlInputs(const std::shared_ptr<session::Context> &context,
                                      const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) {
  return true;
}
bool KernelAdjust::NeedInsertSwitch() { return true; }
void KernelAdjust::Profiling(NotNull<session::KernelGraph *> kernel_graph_ptr) { return; }
}  // namespace device
}  // namespace mindspore
