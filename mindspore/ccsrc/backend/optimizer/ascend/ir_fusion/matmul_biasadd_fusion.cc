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
#include "backend/optimizer/ascend/ir_fusion/matmul_biasadd_fusion.h"
#include <vector>
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace opt {
const BaseRef MatmulBiasaddFusion::DefinePattern() const {
  VectorRef matmul({matmul_var_, x0_, x1_});
  VectorRef pattern({prim::kPrimBiasAdd, matmul, x2_});
  return pattern;
}

const AnfNodePtr MatmulBiasaddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);

  auto matmul = GetAnfNodeByVar(equiv, matmul_var_);
  if (matmul == nullptr || !matmul->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Get CNode MatMul failed!"
                      << " trace: " << trace::DumpSourceLines(node);
  }

  // If there is a side-effect operator in the fusion, do not merge
  if (!IsStateEquivalent(node, matmul)) {
    return node;
  }

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimMatMul->name())));
  inputs.emplace_back(GetAnfNodeByVar(equiv, x0_));
  inputs.emplace_back(GetAnfNodeByVar(equiv, x1_));
  inputs.emplace_back(GetAnfNodeByVar(equiv, x2_));
  auto new_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(node->scope());
  new_node->set_abstract(node->abstract());

  AnfAlgo::CopyNodeAttrs(matmul, new_node);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
