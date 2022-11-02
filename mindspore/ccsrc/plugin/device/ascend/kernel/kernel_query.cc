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

#include "plugin/device/ascend/kernel/kernel_query.h"
#include <algorithm>
#include <string>
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_metadata.h"
#include "plugin/device/ascend/kernel/host/host_kernel_metadata.h"
#include "plugin/device/ascend/kernel/rts/rt_kernel_info.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_select.h"
#include "plugin/device/ascend/kernel/akg/akg_kernel_metadata.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
namespace {
void FilterInvalidKernelInfo(const CNodePtr &kernel_node,
                             std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  if (kernel_info_list->empty()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t output_tensor_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  size_t input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> filtered_list;
  (void)std::copy_if(
    kernel_info_list->begin(), kernel_info_list->end(), std::back_inserter(filtered_list),
    [output_tensor_num, input_tensor_num](const std::shared_ptr<kernel::KernelBuildInfo> &kernel_build_info) {
      MS_EXCEPTION_IF_NULL(kernel_build_info);
      return kernel_build_info->GetOutputNum() == output_tensor_num &&
             kernel_build_info->GetInputNum() == input_tensor_num;
    });
  if (!filtered_list.empty()) {
    kernel_info_list->clear();
    (void)std::copy(filtered_list.begin(), filtered_list.end(), std::back_inserter(*kernel_info_list));
  } else {
    for (size_t index = 0; index < kernel_info_list->size(); ++index) {
      std::ostringstream buffer;
      auto &kernel_info = kernel_info_list->at(index);
      MS_EXCEPTION_IF_NULL(kernel_info);
      if (kernel_info->GetOutputNum() != output_tensor_num) {
        buffer << "Kernel node's output size [" << output_tensor_num << "]"
               << " cannot match the kernel's output size [" << kernel_info->GetOutputNum() << "]";
      } else {
        buffer << "Kernel node's input size [" << input_tensor_num << "]"
               << " cannot match the kernel's input size [" << kernel_info->GetInputNum() << "]";
      }
      MS_LOG(INFO) << "Kernel [ " << index << " ] :" << kernel_info->ToString() << buffer.str();
    }
    kernel_info_list->clear();
    MS_LOG(INFO) << "Node: " << kernel_node->DebugString() << "'s output size : [" << output_tensor_num << "]"
                 << "input size : [" << input_tensor_num << "] can not match any kernelInfo !";
  }
}

bool SelectAicpuReshapeInTaskSink(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (common::AnfAlgo::GetCNodeName(kernel_node) != "Reshape") {
    return false;
  }
  const size_t AicpuReshapeSize = 2;
  if (kernel_node->size() != AicpuReshapeSize) {
    return false;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  return is_task_sink;
}
}  // namespace

void CheckKernelInfoListEmpty(const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list,
                              const std::string &type) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  if (kernel_info_list->empty()) {
    MS_LOG(INFO) << "Warning: kernel info list is empty, kernel type: " << type;
  }
}

void KernelQueryAll(const CNodePtr &kernel_node,
                    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  TbeMetadataInfo(kernel_node, kernel_info_list);
  if (kernel_info_list->empty()) {
    GetRtKelInfo(kernel_node, kernel_info_list);
    CheckKernelInfoListEmpty(kernel_info_list, "RT_Kernel");
  }
  if (kernel_info_list->empty()) {
    HcclMetadataInfo(kernel_node, kernel_info_list);
    CheckKernelInfoListEmpty(kernel_info_list, "HCCL_Kernel");
  }
  if (SelectAicpuReshapeInTaskSink(kernel_node)) {
    return;
  }
  if (kernel_info_list->empty()) {
    HostMetadataInfo(kernel_node, kernel_info_list);
    CheckKernelInfoListEmpty(kernel_info_list, "HOST_Kernel");
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

  const PrimitivePtr kPrimLoadIm2Col = std::make_shared<Primitive>("LoadIm2Col");
  if (IsPrimitiveCNode(kernel_node, kPrimLoadIm2Col)) {
    kernel_type = KernelType::AKG_KERNEL;
  }  // use LoadIm2Col only for THOR optimizer

  switch (kernel_type) {
    case KernelType::AKG_KERNEL:
      AkgMetadataInfo(kernel_node, kernel_info_list);
      break;
    default:
      KernelQueryAll(kernel_node, kernel_info_list);
      break;
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
}  // namespace kernel
}  // namespace mindspore
