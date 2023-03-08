/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/diag_part_fission.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {
const BaseRef DiagPartFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto diag_apart_prim = std::make_shared<Primitive>(prim::kPrimDiagPartD->name());
  return VectorRef({diag_apart_prim, Xs});
}

const AnfNodePtr DiagPartFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto diag_part_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(diag_part_cnode);

  auto type = common::AnfAlgo::GetOutputInferDataType(node, kIndex0);
  if (!CheckOpAICoreSupported(type)) {
    MS_LOG(INFO) << "DiagPart fission failed for aicore, check to aicpu.";
    return nullptr;
  }

  constexpr size_t kDiagPartInputNum = 1;
  if (diag_part_cnode->size() != kDiagPartInputNum + 1) {
    MS_LOG(INFO) << "The node " << diag_part_cnode->DebugString() << " is not equal to " << kDiagPartInputNum
                 << " inputs";
    return nullptr;
  }
  auto out_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  auto input_shape = common::AnfAlgo::GetOutputInferShape(diag_part_cnode->inputs()[kIndex1], 0);
  constexpr size_t kDiagPartInputMaxDim = 8;
  if (input_shape.size() > kDiagPartInputMaxDim) {
    MS_EXCEPTION(ValueError) << "For DiagPart, rank of input should be 2, 4, 6 or 8, but got: " << input_shape.size();
  }
  std::vector<AnfNodePtr> new_node_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDiagPartD->name()))};
  auto assist_node = CreateAssistNode(func_graph, diag_part_cnode, out_shape);
  (void)new_node_inputs.insert(new_node_inputs.cend(), diag_part_cnode->inputs().cbegin() + 1,
                               diag_part_cnode->inputs().cend());
  new_node_inputs.push_back(assist_node);
  CNodePtr new_cnode = NewCNode(new_node_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(diag_part_cnode->abstract());
  new_cnode->set_scope(diag_part_cnode->scope());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_node);
    MS_LOG(INFO) << "Add assist tensor for DiagPart op success.";
  }
  return new_cnode;
}
}  // namespace mindspore::opt
