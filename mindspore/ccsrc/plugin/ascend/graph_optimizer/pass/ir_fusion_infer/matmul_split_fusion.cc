/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_split_fusion.h"
#include <vector>
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
std::vector<std::string> MatmulSplitFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret = {prim::kPrimMatMul->name(), prim::kPrimSplitWithSize->name()};
  return ret;
}

const BaseRef MatmulSplitFusion::DefinePattern() const {
  auto matmul_ref = GetMatmulPattern();
  auto split_with_size_ref = GetSplitWithSizePattern(matmul_ref);
  return split_with_size_ref;
}

const AnfNodePtr MatmulSplitFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const {
  if (!IsEnableMatmulSplit()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(graph);
  auto [split_cnode, split_size_node] = GetSplitSizeNode(node);
  auto [matmul_cnode, input_x, input_w, input_trans_a, input_trans_b] = GetMatmulNode(split_cnode);
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == split_cnode->func_graph(), {});
  if (!CheckMatmulSplit(input_x, input_w, input_trans_a, input_trans_b, split_size_node)) {
    return nullptr;
  }
  PrimitivePtr matmul_split_prim = GetMatmulSplitPrimitive(split_size_node);
  AnfNodePtrList matmul_split_inputs = {input_x, input_w, GetReshapeTupleNode(graph)};
  auto matmul_split_cnode =
    GetMatmulSplitCNode(matmul_split_prim, matmul_split_inputs, graph, matmul_cnode, split_cnode);
  return matmul_split_cnode;
}

std::string MatmulSplitFusion::GetFfnSplitPriName() const { return kMatmulFfnSplitPrimName; }

std::string MatmulSplitFusion::GetQkvSplitPriName() const { return kMatmulQkvSplitPrimName; }

void MatmulSplitFusion::SetMatmulSplitPrimitiveAttr(const PrimitivePtr &matmul_split_prim,
                                                    const ValueNodePtr &split_size_node) const {
  MS_EXCEPTION_IF_NULL(matmul_split_prim);
  MS_EXCEPTION_IF_NULL(split_size_node);
  matmul_split_prim->AddAttr(kNLength, split_size_node->value());
}
}  // namespace opt
}  // namespace mindspore
