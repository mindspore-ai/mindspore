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

#include "runtime/device/common_somas_allocator.h"
#include <utility>
#include <string>
#include "include/backend/optimizer/helper.h"
#include "utils/ms_context.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/string_recorder.h"
#endif

namespace mindspore {
namespace device {
bool CommonSomasAllocator::Assign(const session::KernelGraph &graph) {
  somas::SomasPtr somas_ptr{nullptr};
  if (GetTargetFromContext() == kAscendDevice) {
    somas_ptr = somas::SomasManager::Instance().GetSomas(DeviceType::kAscend);
  } else if (GetTargetFromContext() == kGPUDevice) {
    somas_ptr = somas::SomasManager::Instance().GetSomas(DeviceType::kGPU);
  } else {
    somas_ptr = somas::SomasManager::Instance().GetSomas(DeviceType::kCPU);
  }
  MS_EXCEPTION_IF_NULL(somas_ptr);
  bool ret = somas_ptr->Assign(graph);
  if (ret) {
#ifdef ENABLE_DUMP_IR
    SubModuleId module = SubModuleId::SM_OPTIMIZER;
    std::string name = "somas_allocate_info." + std::to_string(graph.graph_id());
    (void)mindspore::RDR::RecordString(module, name, somas_ptr->SomasInfo());
#endif
#ifndef ENABLE_SECURITY
    somas_ptr->ConvertToProfilingNode(graph.graph_id());
#endif
  }
  return ret;
}

uint8_t *CommonSomasAllocator::GetNodeOutputPtr(const AnfNodePtr &node, size_t index) const {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (index >= kernel_info->somas_output_result().size()) {
    MS_LOG(EXCEPTION) << "index:[" << index << "] is larger than it's output size:["
                      << kernel_info->somas_output_result().size() << "]";
  }
  auto somas_offset_aligned_size = kernel_info->somas_output_result()[index];
  if (somas_offset_aligned_size.second == 0) {
    return nullptr;
  }
  auto somas_offset = somas_offset_aligned_size.first;
  uint8_t *ptr = mem_base_addr_ + somas_offset;
  return ptr;
}

uint8_t *CommonSomasAllocator::GetNodeWorkSpacePtr(const AnfNodePtr &node, size_t index) const {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (index >= kernel_info->somas_workspace_result().size()) {
    MS_LOG(EXCEPTION) << "index:[" << index << "] is larger than it's output size:["
                      << kernel_info->somas_workspace_result().size() << "]";
  }
  auto somas_offset_aligned_size = kernel_info->somas_workspace_result()[index];
  if (somas_offset_aligned_size.second == 0) {
    return nullptr;
  }
  auto somas_offset = somas_offset_aligned_size.first;
  uint8_t *ptr = mem_base_addr_ + somas_offset;
  return ptr;
}
}  // namespace device
}  // namespace mindspore
