/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/matmul_biasadd_fusion.h"
#include <vector>
#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
AnfNodePtr MatmulBiasaddFusion::CreateMatmulWithBias(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &equiv) const {
  auto matmul = GetAnfNodeByVar(equiv, matmul_var_);
  if (matmul == nullptr || !matmul->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Get CNode MatMul failed!" << trace::DumpSourceLines(node);
  }

  // If there is a side-effect operator in the fusion, do not merge
  if (!IsStateEquivalent(node, matmul)) {
    return node;
  }

  std::vector<AnfNodePtr> inputs;
  (void)inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimMatMul->name())));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x0_));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x1_));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x2_));
  auto new_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(node->scope());
  new_node->set_abstract(node->abstract());
  common::AnfAlgo::CopyNodeAttrs(matmul, new_node);
  return new_node;
}

const BaseRef MatmulBiasaddFusion::DefinePattern() const {
  VectorRef matmul({matmul_var_, x0_, x1_});
  VectorRef pattern({prim::kPrimBiasAdd, matmul, x2_});
  return pattern;
}

const AnfNodePtr MatmulBiasaddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);

  return CreateMatmulWithBias(graph, node, equiv);
}

bool MatmulAddFusion::NeedFusion(const AnfNodePtr &add) const {
  auto bias_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(add, kIndex1);
  if (bias_shape.size() != 1) {
    return false;
  }
  return true;
}

const BaseRef MatmulAddFusion::DefinePattern() const {
  VectorRef matmul({matmul_var_, x0_, x1_});
  VectorRef pattern({prim::kPrimAdd, matmul, x2_});
  return pattern;
}

const AnfNodePtr MatmulAddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (!NeedFusion(node)) {
    return nullptr;
  }

  return CreateMatmulWithBias(graph, node, equiv);
}
}  // namespace opt
}  // namespace mindspore
