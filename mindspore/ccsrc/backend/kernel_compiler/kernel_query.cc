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

#include "backend/kernel_compiler/kernel_query.h"
#include <memory>
#include <algorithm>
#include "backend/kernel_compiler/aicpu/aicpu_kernel_metadata.h"
#include "backend/kernel_compiler/host/host_kernel_metadata.h"
#include "backend/kernel_compiler/rts/rt_kernel_info.h"
#include "backend/kernel_compiler/hccl/hccl_kernel_metadata.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_select/tbe_kernel_select.h"
#include "backend/kernel_compiler/akg/akg_kernel_metadata.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
namespace {
void FilterInvalidKernelInfo(const CNodePtr &kernel_node,
                             std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t output_tensor_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  size_t input_tensor_num = AnfAlgo::GetInputTensorNum(kernel_node);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> filtered_list;
  (void)std::copy_if(
    kernel_info_list->begin(), kernel_info_list->end(), std::back_inserter(filtered_list),
    [output_tensor_num, input_tensor_num](const std::shared_ptr<kernel::KernelBuildInfo> &kernel_build_info) {
      return kernel_build_info->GetOutputNum() == output_tensor_num &&
             kernel_build_info->GetInputNum() == input_tensor_num;
    });
  if (!filtered_list.empty()) {
    kernel_info_list->clear();
    (void)std::copy(filtered_list.begin(), filtered_list.end(), std::back_inserter(*kernel_info_list));
  } else {
    MS_LOG(INFO) << "All kernel Info list does not match any kernel info ";
    for (size_t index = 0; index < kernel_info_list->size(); ++index) {
      std::ostringstream buffer;
      auto &kernel_info = kernel_info_list->at(index);
      MS_EXCEPTION_IF_NULL(kernel_info);
      if (kernel_info->GetOutputNum() != output_tensor_num) {
        buffer << "Kernel node's output size [" << output_tensor_num << "]"
               << " cannot match the kernel's output size [" << kernel_info->GetOutputNum() << "]";
      } else {
        buffer << "Kernel node's output size [" << input_tensor_num << "]"
               << " cannot match the kernel's output size [" << kernel_info->GetInputNum() << "]";
      }
      MS_LOG(INFO) << "kernel [ " << index << " ] :" << kernel_info->ToString() << buffer.str();
    }
    kernel_info_list->clear();
    MS_LOG(INFO) << "node" << kernel_node->DebugString() << "'s output size : [" << output_tensor_num << "]"
                 << "input size : [" << input_tensor_num << "] cannot match any kernelInfo !";
  }
}
}  // namespace

void KernelQueryAll(const CNodePtr &kernel_node,
                    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);
  TbeMetadataInfo(kernel_node, kernel_info_list);
  if (kernel_info_list->empty()) {
    AicpuMetadataInfo(kernel_node, kernel_info_list);
    if (!kernel_info_list->empty()) {
      MS_LOG(INFO) << "The node [" << kernel_node->DebugString()
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
    HostMetadataInfo(kernel_node, kernel_info_list);
  }
  if (kernel_info_list->empty()) {
    MS_EXCEPTION(NotExistsError)
      << "Failed to obtain operator info, Please check whether the operator info is registered, Op full name:"
      << kernel_node->fullname_with_scope() << "Node Type: " << op_name
      << ", Node DebugString: " << kernel_node->DebugString() << "\n trace: " << trace::DumpSourceLines(kernel_node);
  }
}

void KernelQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list,
                 KernelType kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

  const PrimitivePtr kPrimProdForceSeA = std::make_shared<Primitive>("ProdForceSeA");
  if (IsPrimitiveCNode(kernel_node, kPrimProdForceSeA)) {
    kernel_type = KernelType::AKG_KERNEL;
  }
  switch (kernel_type) {
    case KernelType::AKG_KERNEL:
      AkgMetadataInfo(kernel_node, kernel_info_list);
      break;
    default:
      KernelQueryAll(kernel_node, kernel_info_list);
      break;
  }

  if (kernel_info_list->empty()) {
    MS_EXCEPTION(NotExistsError)
      << "Failed to obtain operator info. Please check whether the operator info is registered, Op full name:"
      << kernel_node->fullname_with_scope() << ". Node DebugString: " << kernel_node->DebugString();
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
                       return item->IsSimilarityKernelBuildInfo(*select_kernel_build_info);
                     });
}

bool IsSupportedByAICore(const AnfNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(select_kernel_build_info);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto cnode = kernel_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  TbeMetadataInfo(cnode, &kernel_info_list);
  return std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                     [&select_kernel_build_info](const kernel::KernelBuildInfoPtr item) {
                       MS_EXCEPTION_IF_NULL(item);
                       return *item == *select_kernel_build_info;
                     });
}
}  // namespace kernel
}  // namespace mindspore
