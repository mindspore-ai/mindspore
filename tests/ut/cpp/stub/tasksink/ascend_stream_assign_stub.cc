/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/hal/device/ascend_label_assign.h"
#include "plugin/device/ascend/hal/device/kernel_adjust.h"

namespace mindspore {
namespace device {
namespace ascend {

void AscendLabelAssign::AssignLabel(NotNull<std::shared_ptr<session::KernelGraph>> graph) {}
uint32_t AscendLabelAssign::GetLabelNum(NotNull<const session::KernelGraph *> graph) { return 1; }
uint32_t AscendLabelAssign::GetLabelNum(NotNull<std::shared_ptr<session::KernelGraph>> graph) { return 1; }
void AscendStreamAssign::AssignStream(const NotNull<KernelGraphPtr> &graph_ptr) { return; }

void AscendStreamAssign::GetWaitStreams(vector<uint32_t> *wait_active_stream_list) { return; }

void AscendStreamAssign::GetHcomStreams(std::vector<uint32_t> *streams) { return; }

void AscendStreamAssign::AssignStreamForNonTaskSink(const std::vector<CNodePtr> &kernels) { return; }

uint32_t AscendStreamAssign::GetHcomTaskNum(const CNodePtr &) { return 200; }
}  // namespace ascend

void KernelAdjust::InsertDeviceLoopCtrl(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const { return; }
void KernelAdjust::AssignLoopCtrlMemory(const session::KernelGraph &kernel_graph_ptr) const { return; }
void KernelAdjust::LoadDeviceLoopCtrlParameters(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  return;
}

bool KernelAdjust::NeedLoopSink() { return true; }

void KernelAdjust::ProcessLoopSink(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const { return; }

#ifndef ENABLE_SECURITY
void KernelAdjust::Profiling(NotNull<session::KernelGraph *> kernel_graph_ptr) { return; }
#endif
void KernelAdjust::InsertOverflowCheckOperations(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr) const {
  return;
}
}  // namespace device
}  // namespace mindspore
