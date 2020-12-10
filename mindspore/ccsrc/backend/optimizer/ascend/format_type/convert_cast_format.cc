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
#include "backend/optimizer/ascend/format_type/convert_cast_format.h"

#include <memory>

#include "backend/session/anf_runtime_algorithm.h"
namespace mindspore {
namespace opt {
const BaseRef ConvertCastFormat::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

const AnfNodePtr ConvertCastFormat::Process(const mindspore::FuncGraphPtr &, const mindspore::AnfNodePtr &node,
                                            const mindspore::EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto node_name = AnfAlgo::GetCNodeName(node);
  if (node_name == prim::kPrimCast->name()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(cnode); ++input_index) {
    auto input_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(cnode, input_index), 0).first;
    MS_EXCEPTION_IF_NULL(input_node);
    if (!input_node->isa<CNode>()) {
      continue;
    }
    auto cast_node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cast_node);
    auto input_node_name = AnfAlgo::GetCNodeName(cast_node);
    if (input_node_name != prim::kPrimCast->name()) {
      continue;
    }
    auto format = AnfAlgo::GetInputFormat(node, input_index);
    auto cast_input_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(cast_node, 0), 0).first;
    auto cast_input_format = AnfAlgo::GetOutputFormat(cast_input_node, 0);
    // change cast to default that can be more faster when it cast other hw format
    if (cast_input_format != format) {
      if (cast_input_format == kOpFormat_DEFAULT || format == kOpFormat_DEFAULT) {
        auto info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(
          AnfAlgo::GetSelectKernelBuildInfo(cast_node));
        info_builder->SetInputsFormat({kOpFormat_DEFAULT});
        info_builder->SetOutputsFormat({kOpFormat_DEFAULT});
        AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), cast_node.get());
      }
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
