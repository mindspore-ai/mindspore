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
#include "kernel/mng/rt_kernel_info.h"
#include "kernel/hccl/hccl_kernel_metadata.h"
#include "kernel/tbe/tbe_kernel_select.h"
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
  kernel_info_list->clear();
  if (!filtered_list.empty()) {
    (void)std::copy(filtered_list.begin(), filtered_list.end(), std::back_inserter(*kernel_info_list));
  } else {
    MS_LOG(EXCEPTION) << "node" << kernel_node->DebugString() << "'s output size : ["
                      << AnfAlgo::GetOutputTensorNum(kernel_node) << "]"
                      << "input size : [" << AnfAlgo::GetInputTensorNum(kernel_node)
                      << "] cannot match any kernelInfo !";
  }
}
}  // namespace
void KernelQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  TbeMetadataInfo(kernel_node, kernel_info_list);

  if (kernel_info_list->empty()) {
    AicpuMetadataInfo(kernel_node, kernel_info_list);
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
  FilterInvalidKernelInfo(kernel_node, kernel_info_list);
}
}  // namespace kernel
}  // namespace mindspore
