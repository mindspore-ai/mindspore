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

#include "kernel/kernel_query.h"
#include <memory>
#include <algorithm>
#include "kernel/aicpu/aicpu_kernel_metadata.h"
#include "kernel/rts/rt_kernel_info.h"
#include "kernel/hccl/hccl_kernel_metadata.h"
#include "kernel/tbe/tbe_kernel_select.h"
#include "kernel/akg/akg_kernel_metadata.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
namespace {
void FilterInvalidKernelInfo(const CNodePtr &kernel_node,
                             std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> filtered_list;
  (void)std::copy_if(kernel_info_list->begin(), kernel_info_list->end(), std::back_inserter(filtered_list),
                     [&](const std::shared_ptr<kernel::KernelBuildInfo> &kernel_build_info) {
                       return AnfAlgo::GetOutputTensorNum(kernel_node) == kernel_build_info->GetOutputNum() &&
                              AnfAlgo::GetInputTensorNum(kernel_node) == kernel_build_info->GetInputNum();
                     });
  if (!filtered_list.empty()) {
    kernel_info_list->clear();
    (void)std::copy(filtered_list.begin(), filtered_list.end(), std::back_inserter(*kernel_info_list));
  } else {
    MS_LOG(WARNING) << "All kernel Info list does not match any kernel info ";
    for (size_t index = 0; index < kernel_info_list->size(); ++index) {
      MS_EXCEPTION_IF_NULL(kernel_info_list->at(index));
      MS_LOG(WARNING) << "kernel [ " << index << " ] :" << kernel_info_list->at(index)->ToString();
    }
    MS_LOG(WARNING) << "node" << kernel_node->DebugString() << "'s output size : ["
                    << AnfAlgo::GetOutputTensorNum(kernel_node) << "]"
                    << "input size : [" << AnfAlgo::GetInputTensorNum(kernel_node) << "] cannot match any kernelInfo !";
  }
}
}  // namespace

void KernelQueryAll(const CNodePtr &kernel_node,
                    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);

  TbeMetadataInfo(kernel_node, kernel_info_list);

  if (kernel_info_list->empty()) {
    AicpuMetadataInfo(kernel_node, kernel_info_list);
    if (!kernel_info_list->empty()) {
      MS_LOG(INFO) << "Warning The node [" << kernel_node->DebugString()
                   << "] cannot find valid TBE kernel info, try to get aicpu kernel info";
      AnfAlgo::SetNodeAttr(kAttrIsAICPUKernel, MakeValue(true), kernel_node);
    }
  }

  if (kernel_info_list->empty()) {
    GetRtKelInfo(kernel_node, kernel_info_list);
  }

  if (kernel_info_list->empty()) {
    HcclMetadataInfo(kernel_node, kernel_info_list);
  }
  if (kernel_info_list->empty()) {
    MS_LOG(EXCEPTION) << "Op " << kernel_node->DebugString() << "kernel query fail!";
  }
}

void KernelQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list,
                 KernelType kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);

  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);

  switch (kernel_type) {
    case KernelType::AKG_KERNEL:
      AkgMetadataInfo(kernel_node, kernel_info_list);
      break;
    default:
      KernelQueryAll(kernel_node, kernel_info_list);
      break;
  }

  if (kernel_info_list->empty()) {
    MS_EXCEPTION(NotExistsError) << "Op[" << kernel_node->DebugString() << "] kernel query fail!";
  }
  // check output
  FilterInvalidKernelInfo(kernel_node, kernel_info_list);
}

void AICPUQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  kernel_info_list->clear();
  AicpuMetadataInfo(kernel_node, kernel_info_list);
  FilterInvalidKernelInfo(kernel_node, kernel_info_list);
}
bool IsSupportedByAICPU(const AnfNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(select_kernel_build_info);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto cnode = kernel_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AICPUQuery(cnode, &kernel_info_list);
  return std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                     [&select_kernel_build_info](const kernel::KernelBuildInfoPtr item) {
                       MS_EXCEPTION_IF_NULL(item);
                       return *item == *select_kernel_build_info;
                     });
}

bool IsSupportedByAICore(const AnfNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(select_kernel_build_info);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto cnode = kernel_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  TbeMetadataInfo(cnode, &kernel_info_list);
  FilterInvalidKernelInfo(cnode, &kernel_info_list);
  return std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                     [&select_kernel_build_info](const kernel::KernelBuildInfoPtr item) {
                       MS_EXCEPTION_IF_NULL(item);
                       return *item == *select_kernel_build_info;
                     });
}
}  // namespace kernel
}  // namespace mindspore
