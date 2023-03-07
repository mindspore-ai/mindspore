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

#include "plugin/device/ascend/optimizer/format_type/convert_unsupported_transnode_to_aicpu.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace opt {
const BaseRef ConvertUnSupportNodeToAICPU::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

const AnfNodePtr ConvertUnSupportNodeToAICPU::Process(const mindspore::FuncGraphPtr &graph,
                                                      const mindspore::AnfNodePtr &node,
                                                      const mindspore::EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->has_flag(kAttrMutableKernel)) {
    return nullptr;
  }
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  if (node_name != prim::kPrimTransData->name() && node_name != prim::kPrimCast->name()) {
    return nullptr;
  }
  auto kernel_builder_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  if (CheckAICoreSupportedSpec(node, kernel_builder_info)) {
    return nullptr;
  } else if (CheckAICPUSupportedSpec(node, kernel_builder_info)) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(kernel_builder_info);
    MS_EXCEPTION_IF_NULL(builder);
    builder->SetKernelType(AICPU_KERNEL);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
    common::AnfAlgo::SetNodeAttr(kAttrIsAiCpuKernel, MakeValue(true), node);
  } else {
    MS_LOG(EXCEPTION) << "Kernel " << kernel_builder_info->ToString() << "is not supported in AiCPU & AiCore : node ["
                      << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
