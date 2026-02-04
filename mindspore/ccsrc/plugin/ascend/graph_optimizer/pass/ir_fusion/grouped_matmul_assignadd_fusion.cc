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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/grouped_matmul_assignadd_fusion.h"
#include <vector>
#include <string>
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "primitive/nn_optimizer_op_name.h"
#include "primitive/array_op_name.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace opt {
GroupedMatmulAssignaddFusion::GroupedMatmulAssignaddFusion(bool multigraph)
    : PatternProcessPass("grouped_matmul_assignadd_fusion", multigraph) {
  x_ = std::make_shared<Var>();
  weight_ = std::make_shared<Var>();
  group_list_ = std::make_shared<Var>();
  split_item_ = std::make_shared<Var>();
  group_type_ = std::make_shared<Var>();
  out_ = std::make_shared<Var>();
  transpose_a_ = std::make_shared<Var>();
  transpose_b_ = std::make_shared<Var>();
  grouped_matmul_ = std::make_shared<Var>(prim::kPrimGroupedMatmul);
}

bool GroupedMatmulAssignaddFusion::CheckFusion(const CNodePtr &grouped_matmul, const EquivPtr &equiv) const {
  // check split_item=3, groupe_type=2
  auto split_item = GetAnfNodeByVar(equiv, split_item_);
  auto group_type = GetAnfNodeByVar(equiv, group_type_);
  if (!split_item->isa<ValueNode>() || !group_type->isa<ValueNode>()) {
    MS_LOG(INFO) << "Expect split_item and group_type input should be ValueNode, but not, skip fusion.";
    return false;
  }
  auto split_item_value = split_item->cast<ValueNodePtr>()->value();
  auto group_type_value = group_type->cast<ValueNodePtr>()->value();
  if (!split_item_value->isa<Int64Imm>() || !group_type_value->isa<Int64Imm>()) {
    MS_LOG(INFO) << "Expect the datatype of split_item and group_type input should be int64, but not, skip fusion.";
    return false;
  }
  auto split_item_int = GetValue<int64_t>(split_item_value);
  auto group_type_int = GetValue<int64_t>(group_type_value);
  const int64_t split_item_type3 = 3;
  const int64_t group_type_k = 2;
  if (split_item_int != split_item_type3 || group_type_int != group_type_k) {
    MS_LOG(INFO) << "Expect split_item is " << split_item_type3 << " and groupe_type is " << group_type_k
                 << ", but got split_item[" << split_item_int << "] and group_type[" << group_type_int
                 << "], skip fusion.";
    return false;
  }

  // check transpose_a=false, transpose_b=false
  auto transpose_a = GetAnfNodeByVar(equiv, transpose_a_);
  auto transpose_b = GetAnfNodeByVar(equiv, transpose_b_);
  if (!transpose_a->isa<ValueNode>() || !transpose_b->isa<ValueNode>()) {
    MS_LOG(INFO) << "Expect transpose_a and transpose_b input should be ValueNode, but not, skip fusion.";
    return false;
  }
  auto transpose_a_value = transpose_a->cast<ValueNodePtr>()->value();
  auto transpose_b_value = transpose_b->cast<ValueNodePtr>()->value();
  if (!transpose_a_value->isa<BoolImm>() || !transpose_b_value->isa<BoolImm>()) {
    MS_LOG(INFO) << "Expect the datatype of transpose_a and transpose_b input should be bool, but not, skip fusion.";
    return false;
  }
  auto transpose_a_bool = GetValue<bool>(transpose_a_value);
  auto transpose_b_bool = GetValue<bool>(transpose_b_value);
  if (transpose_a_bool != false || transpose_b_bool != false) {
    MS_LOG(INFO) << "Expect transpose_a and transpose_b input is false, but got transpose_a[" << transpose_a_bool
                 << "] and transpose_b[" << transpose_b_bool << "], skip fusion.";
    return false;
  }

  // check 3rd to 7th input is None
  const size_t none_start_idx = 2;
  const size_t none_end_idx = 6;
  for (size_t idx = none_start_idx; idx <= none_end_idx; ++idx) {
    auto input_node = common::AnfAlgo::GetInputNode(grouped_matmul, idx);
    if (!input_node->isa<ValueNode>() || !input_node->cast<ValueNodePtr>()->value()->isa<None>()) {
      MS_LOG(INFO) << "Expect GroupedMatmul's 3rd to 7th input is None, but not, skip fusion.";
      return false;
    }
  }

  return true;
}

bool GroupedMatmulAssignaddFusion::CheckDataType(const AnfNodePtr &input_x, const AnfNodePtr &weight,
                                                 const AnfNodePtr &group_list, const AnfNodePtr &out) const {
  auto input_x_dtype = common::AnfAlgo::GetOutputInferDataType(input_x, 0);
  auto weight_dtype = common::AnfAlgo::GetOutputInferDataType(weight, 0);
  auto group_list_dtype = common::AnfAlgo::GetOutputInferDataType(group_list, 0);
  auto out_dtype = common::AnfAlgo::GetOutputInferDataType(out, 0);
  if (input_x_dtype != kNumberTypeFloat16 && input_x_dtype != kNumberTypeBFloat16) {
    MS_LOG(INFO) << "Input x's datatype is not float16 or bfloat16, skip fusion.";
    return false;
  }
  if (weight_dtype != kNumberTypeFloat16 && weight_dtype != kNumberTypeBFloat16) {
    MS_LOG(INFO) << "Input weight's datatype is not float16 or bfloat16, skip fusion.";
    return false;
  }
  if (group_list_dtype != kNumberTypeInt64) {
    MS_LOG(INFO) << "Input group_list's datatype is not Int64, skip fusion.";
    return false;
  }
  if (out_dtype != kNumberTypeFloat32) {
    MS_LOG(INFO) << "Input out's datatype is not float32, skip fusion.";
    return false;
  }
  return true;
}

void GroupedMatmulAssignaddFusion::ReplaceGMMForDepend(const FuncGraphPtr &graph, const CNodePtr &gmm,
                                                       const CNodePtr &gmm_add) const {
  // replace GMM with GMMAdd when GMM is depend's 2nd input.(depend is created by dw masking)
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &gmm_users = manager->node_users()[gmm];
  for (const auto &[node, index] : gmm_users) {
    const int depend_idx = 2;
    if (IsPrimitiveCNode(node, prim::kPrimDepend) && index == depend_idx) {
      (void)manager->SetEdge(node, kIndex2, gmm_add);
    }
  }
}

std::vector<std::string> GroupedMatmulAssignaddFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{kTransposeExtViewOpName, prim::kPrimGroupedMatmul->name(), kAssignAddOpName};
  return ret;
}

const BaseRef GroupedMatmulAssignaddFusion::DefinePattern() const {
  VarPtr transpose_dim1 = std::make_shared<Var>();
  VarPtr transpose_dim2 = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr getitem_index = std::make_shared<Var>();
  auto transpose =
    VectorRef({std::make_shared<Primitive>(kTransposeExtViewOpName), x_, transpose_dim1, transpose_dim2});
  auto maketuple = VectorRef({prim::kPrimMakeTuple, transpose});
  auto grouped_matmul = VectorRef(
    {grouped_matmul_, maketuple, weight_, Xs, group_list_, split_item_, group_type_, transpose_a_, transpose_b_});
  auto getitem = VectorRef({prim::kPrimTupleGetItem, grouped_matmul, getitem_index});
  VarPtr monad = std::make_shared<SeqVar>();
  auto assignadd = VectorRef({std::make_shared<Primitive>(kAssignAddOpName), out_, getitem, monad});
  return assignadd;
}

const AnfNodePtr GroupedMatmulAssignaddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                       const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto assign_add = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(assign_add);

  if (AnfAlgo::GetBackend(graph) != kBackendMSBackend) {
    return nullptr;
  }
  auto grouped_matmul = GetAnfNodeByVar(equiv, grouped_matmul_);
  auto grouped_matmul_cnode = grouped_matmul->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(grouped_matmul_cnode);
  if (!CheckFusion(grouped_matmul_cnode, equiv)) {
    return nullptr;
  }

  auto input_x = GetAnfNodeByVar(equiv, x_);
  auto weight = GetAnfNodeByVar(equiv, weight_);
  auto group_list = GetAnfNodeByVar(equiv, group_list_);
  auto out = GetAnfNodeByVar(equiv, out_);
  auto weight_node = CreatTupleGetItemNode(graph, weight, 0);
  if (!CheckDataType(input_x, weight_node, group_list, out)) {
    return nullptr;
  }

  // create InplaceGroupedMatmulAdd
  const std::string fusion_op_name = "InplaceGroupedMatmulAdd";
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(fusion_op_name)), input_x, weight_node,
                                    group_list, out};
  const size_t assign_add_monad_idx = 3;
  if (common::AnfAlgo::GetInputNum(assign_add) == assign_add_monad_idx) {
    // add monad input
    (void)inputs.emplace_back(assign_add->input(assign_add_monad_idx));
  }
  auto grouped_matmul_add = NewCNode(inputs, graph);
  grouped_matmul_add->set_scope(grouped_matmul->scope());
  grouped_matmul_add->set_abstract(node->abstract());
  if (common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, assign_add)) {
    common::AnfAlgo::CopyNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, assign_add, grouped_matmul_add);
  }
  ReplaceGMMForDepend(graph, grouped_matmul_cnode, grouped_matmul_add);
  return grouped_matmul_add;
}
}  // namespace opt
}  // namespace mindspore
