/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/assign_add_opt.h"
#include <memory>
#include <vector>
#include <list>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kAttrConcatN = "N";

void MergeMultiMatmulAssingAdd(const FuncGraphManagerPtr &manager, const FuncGraphPtr &each_graph,
                               const std::vector<CNodePtr> &matmul_dw_nodes,
                               const std::pair<AnfNodePtr, std::vector<AnfNodePtr>> &pair) {
  auto matmul_dw_nodes_front = matmul_dw_nodes.front();
  // concat all matmul nodes inputs
  std::vector<AnfNodePtr> maketuple1_inputs{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AnfNodePtr> maketuple2_inputs{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AbstractBasePtr> maketuple1_abs_inputs;
  std::vector<AbstractBasePtr> maketuple2_abs_inputs;
  std::transform(matmul_dw_nodes.begin(), matmul_dw_nodes.end(), std::back_inserter(maketuple1_inputs),
                 [](CNodePtr anfnode) { return anfnode->input(1); });
  std::transform(matmul_dw_nodes.begin(), matmul_dw_nodes.end(), std::back_inserter(maketuple2_inputs),
                 [](CNodePtr anfnode) { return anfnode->input(2); });
  std::transform(matmul_dw_nodes.begin(), matmul_dw_nodes.end(), std::back_inserter(maketuple1_abs_inputs),
                 [](CNodePtr anfnode) { return anfnode->input(1)->abstract()->Clone(); });
  std::transform(matmul_dw_nodes.begin(), matmul_dw_nodes.end(), std::back_inserter(maketuple2_abs_inputs),
                 [](CNodePtr anfnode) { return anfnode->input(2)->abstract()->Clone(); });
  auto maketuple1 = each_graph->NewCNode(maketuple1_inputs);
  auto maketuple2 = each_graph->NewCNode(maketuple2_inputs);
  maketuple1->set_abstract(std::make_shared<abstract::AbstractTuple>(maketuple1_abs_inputs));
  maketuple2->set_abstract(std::make_shared<abstract::AbstractTuple>(maketuple2_abs_inputs));
  std::vector<AnfNodePtr> concat1_inputs{NewValueNode(prim::kPrimConcat), maketuple1};
  std::vector<AnfNodePtr> concat2_inputs{NewValueNode(prim::kPrimConcat), maketuple2};
  auto concat1 = each_graph->NewCNode(concat1_inputs);
  auto concat2 = each_graph->NewCNode(concat2_inputs);
  std::vector<AnfNodePtr> merged_matmul_inputs{NewValueNode(prim::kPrimMatMul), concat1, concat2};
  auto merged_matmul = each_graph->NewCNode(merged_matmul_inputs);
  // set abstract and attr
  auto matmul_dw_node_front = matmul_dw_nodes.front();
  auto mat1_prim = GetCNodePrimitive(matmul_dw_nodes_front);
  auto transpose_a1 = GetValue<bool>(mat1_prim->GetAttr(TRANSPOSE_A));
  auto transpose_b1 = GetValue<bool>(mat1_prim->GetAttr(TRANSPOSE_B));
  auto matmul_dw_node_front_input_node1_abstract = matmul_dw_node_front->input(1)->abstract();
  auto matmul_dw_node_front_input_node2_abstract = matmul_dw_node_front->input(2)->abstract();
  MS_EXCEPTION_IF_NULL(matmul_dw_node_front_input_node1_abstract);
  MS_EXCEPTION_IF_NULL(matmul_dw_node_front_input_node2_abstract);
  auto matmul_dw_node_front_input_node1_input_shape =
    matmul_dw_node_front_input_node1_abstract->BuildShape()->cast<abstract::ShapePtr>()->shape();
  auto matmul_dw_node_front_input_node2_input_shape =
    matmul_dw_node_front_input_node2_abstract->BuildShape()->cast<abstract::ShapePtr>()->shape();
  int64_t axis1 = int64_t(matmul_dw_node_front_input_node1_input_shape.size() - 1);
  if (transpose_a1) {
    axis1 -= 1;
  }
  int64_t axis2 = 0;
  if (transpose_b1) {
    axis2 += 1;
  }
  concat1->set_abstract(matmul_dw_node_front_input_node1_abstract->Clone());
  concat2->set_abstract(matmul_dw_node_front_input_node2_abstract->Clone());
  matmul_dw_node_front_input_node1_input_shape[size_t(axis1)] *= SizeToLong(matmul_dw_nodes.size());
  matmul_dw_node_front_input_node2_input_shape[size_t(axis2)] *= SizeToLong(matmul_dw_nodes.size());
  auto concat1_shape_value = std::make_shared<abstract::Shape>(matmul_dw_node_front_input_node1_input_shape);
  concat1->abstract()->set_shape(concat1_shape_value);
  auto concat2_shape_value = std::make_shared<abstract::Shape>(matmul_dw_node_front_input_node2_input_shape);
  concat2->abstract()->set_shape(concat2_shape_value);
  auto concat1_prim = GetCNodePrimitive(concat1);
  concat1_prim->set_attr(AXIS, MakeValue<int64_t>(axis1));
  concat1_prim->set_attr(kAttrInputNums, MakeValue<int64_t>(maketuple1_abs_inputs.size()));
  concat1_prim->set_attr(kAttrConcatN, MakeValue<int64_t>(maketuple1_abs_inputs.size()));
  auto concat2_prim = GetCNodePrimitive(concat2);
  concat2_prim->set_attr(AXIS, MakeValue<int64_t>(axis2));
  concat2_prim->set_attr(kAttrInputNums, MakeValue<int64_t>(maketuple2_abs_inputs.size()));
  concat2_prim->set_attr(kAttrConcatN, MakeValue<int64_t>(maketuple2_abs_inputs.size()));
  merged_matmul->set_abstract(matmul_dw_node_front->abstract()->Clone());
  auto merged_matmul_prim = GetCNodePrimitive(merged_matmul);
  merged_matmul_prim->SetAttrs(mat1_prim->attrs());
  merged_matmul->set_attrs(matmul_dw_node_front->attrs());
  merged_matmul->set_primal_attrs(matmul_dw_node_front->primal_attrs());
  // concat1 -> merged_matmul -> assign_add
  // concat2
  std::vector<AnfNodePtr> assign_add_inputs{NewValueNode(prim::kPrimAssignAdd), pair.first, merged_matmul};
  auto assign_add_cnode = each_graph->NewCNode(assign_add_inputs);
  assign_add_cnode->set_abstract(merged_matmul->abstract()->Clone());
  for (const auto &assgin_add_origin_node : pair.second) {
    manager->Replace(assgin_add_origin_node, assign_add_cnode);
  }
}
}  // namespace

void AssignAddOpt(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  auto manager = graph->manager();
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> assign_add_map;
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimAssignAdd)) {
        continue;
      }
      assign_add_map[node->input(1)].push_back(node);
    }
    for (const auto &pair : assign_add_map) {
      if (pair.second.size() <= 1) {
        continue;
      }
      auto node_users = manager->node_users()[pair.first];
      if (node_users.size() != pair.second.size()) {
        continue;
      }
      std::vector<CNodePtr> matmul_dw_nodes;
      // Check all input of assignadd node is matmul
      for (const auto &assign_add_node : pair.second) {
        if (IsPrimitiveCNode(assign_add_node->cast<CNodePtr>()->input(2), prim::kPrimMatMul)) {
          auto matmul_node = assign_add_node->cast<CNodePtr>()->input(2)->cast<CNodePtr>();
          if (matmul_node->HasPrimalAttr("DISABLE_MERGE_ASSIGN_ADD")) {
            matmul_dw_nodes.clear();
            break;
          }
          matmul_dw_nodes.push_back(matmul_node);
        } else {
          matmul_dw_nodes.clear();
          break;
        }
      }
      if (matmul_dw_nodes.size() < 2) {
        continue;
      }
      auto matmul_dw_nodes_front = matmul_dw_nodes.front();
      auto is_same_matmul = std::all_of(
        matmul_dw_nodes.begin(), matmul_dw_nodes.end(), [&matmul_dw_nodes_front](const CNodePtr &matmul_dw_node) {
          auto mat1_prim = GetCNodePrimitive(matmul_dw_nodes_front);
          auto mat2_prim = GetCNodePrimitive(matmul_dw_node);
          auto transpose_a1 = GetValue<bool>(mat1_prim->GetAttr(TRANSPOSE_A));
          auto transpose_a2 = GetValue<bool>(mat2_prim->GetAttr(TRANSPOSE_A));
          auto transpose_b1 = GetValue<bool>(mat1_prim->GetAttr(TRANSPOSE_B));
          auto transpose_b2 = GetValue<bool>(mat2_prim->GetAttr(TRANSPOSE_B));
          if (transpose_a1 != transpose_a2 || transpose_b1 != transpose_b2) {
            return false;
          }
          auto input_node_a1 = matmul_dw_nodes_front->input(1);
          auto input_node_a2 = matmul_dw_node->input(1);
          auto input_node_a1_shape = input_node_a1->abstract()->BuildShape();
          MS_EXCEPTION_IF_NULL(input_node_a1_shape);
          MS_EXCEPTION_IF_NULL(input_node_a1_shape->cast<abstract::ShapePtr>());
          auto input_node_a1_shape_element = input_node_a1_shape->cast<abstract::ShapePtr>()->shape();
          auto input_node_a2_shape = input_node_a2->abstract()->BuildShape();
          MS_EXCEPTION_IF_NULL(input_node_a2_shape);
          MS_EXCEPTION_IF_NULL(input_node_a2_shape->cast<abstract::ShapePtr>());
          auto input_node_a2_shape_element = input_node_a2_shape->cast<abstract::ShapePtr>()->shape();
          if (input_node_a1_shape_element != input_node_a2_shape_element) {
            return false;
          }
          auto input_node_b1 = matmul_dw_nodes_front->input(2);
          auto input_node_b2 = matmul_dw_node->input(2);
          auto input_node_b1_shape = input_node_b1->abstract()->BuildShape();
          MS_EXCEPTION_IF_NULL(input_node_b1_shape);
          MS_EXCEPTION_IF_NULL(input_node_b1_shape->cast<abstract::ShapePtr>());
          auto input_node_b1_shape_element = input_node_b1_shape->cast<abstract::ShapePtr>()->shape();
          auto input_node_b2_shape = input_node_b2->abstract()->BuildShape();
          MS_EXCEPTION_IF_NULL(input_node_b2_shape);
          MS_EXCEPTION_IF_NULL(input_node_b2_shape->cast<abstract::ShapePtr>());
          auto input_node_b2_shape_element = input_node_b2_shape->cast<abstract::ShapePtr>()->shape();
          if (input_node_b1_shape_element != input_node_b2_shape_element) {
            return false;
          }
          return true;
        });
      if (!is_same_matmul) {
        return;
      }
      MergeMultiMatmulAssingAdd(manager, each_graph, matmul_dw_nodes, pair);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
