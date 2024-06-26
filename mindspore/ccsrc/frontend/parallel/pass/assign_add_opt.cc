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
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/utils/convert_utils_base.h"
#include "mindspore/ccsrc/pipeline/jit/ps/static_analysis/static_analysis.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kAttrConcatN = "N";
constexpr auto kAttrCastDw = "CastDw";
constexpr auto kAttrTmpMakeTuple = "tmp_make_tuple";
constexpr size_t kPartialPreIndex = 2;

CNodePtr GetMatmulDwNodeFront(const std::vector<CNodePtr> &matmul_dw_nodes,
                              const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map) {
  auto matmul_dw_node_front = matmul_dw_nodes.front();
  for (auto matmul_dw_node : matmul_dw_nodes) {
    auto dx_dw_iter = std::find_if(backward_matmul_dx_dw_map.begin(), backward_matmul_dx_dw_map.end(), [&](auto dx_dw) {
      return (dx_dw.second == matmul_dw_node && dx_dw.first->HasAttr(INTERLEAVED_OVERLAP_MATMUL));
    });
    if (dx_dw_iter != backward_matmul_dx_dw_map.end()) {
      continue;
    }
    matmul_dw_node_front = matmul_dw_node;
    break;
  }
  return matmul_dw_node_front;
}

std::pair<int64_t, int64_t> GetMatMulReduceAxis(size_t matmul_input_shape_size, bool transpose_a1, bool transpose_b1) {
  int64_t axis1 = SizeToLong(matmul_input_shape_size) - 1;
  if (transpose_a1) {
    axis1 -= 1;
  }
  int64_t axis2 = SizeToLong(matmul_input_shape_size) - 2;
  if (transpose_b1) {
    axis2 += 1;
  }
  return {axis1, axis2};
}

void UpdateValueNodeAbs(ValueNodePtr *axis) {
  MS_EXCEPTION_IF_NULL(axis);
  auto value_node = (*axis);
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(value_node->value());
  auto value_abs = value_node->value()->ToAbstract();
  MS_EXCEPTION_IF_NULL(value_abs);
  value_node->set_abstract(value_abs);
}

bool GetMatMulTransposeValue(const CNodePtr &matmul_node, const std::string &attr_name) {
  auto mat_prim = GetCNodePrimitive(matmul_node);
  auto prim_name = mat_prim->name();
  auto &inputs = matmul_node->inputs();
  auto idx = ops::GetInputIndexByName(prim_name, attr_name);
  std::vector<ValuePtr> input_value;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>() || inputs[index]->isa<tensor::Tensor>()) {
      (void)input_value.emplace_back(GetValueNode(inputs[index]));
      continue;
    }
    (void)input_value.emplace_back(nullptr);
  }
  return GetScalarValueFromInputs<bool>(input_value, idx).value();
}

CNodePtr InsertConcat(const std::vector<CNodePtr> &matmul_dw_nodes, const FuncGraphPtr &graph, size_t para_index,
                      const std::vector<AnfNodePtr> &concat_inputs,
                      const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map) {
  std::vector<AnfNodePtr> maketuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AbstractBasePtr> maketuple_abs_inputs;
  (void)std::transform(concat_inputs.begin(), concat_inputs.end(), std::back_inserter(maketuple_inputs),
                       [](AnfNodePtr anf_node) { return anf_node; });
  (void)std::transform(concat_inputs.begin(), concat_inputs.end(), std::back_inserter(maketuple_abs_inputs),
                       [](AnfNodePtr anf_node) { return anf_node->abstract()->Clone(); });
  auto maketuple = graph->NewCNode(maketuple_inputs);
  maketuple->set_abstract(std::make_shared<abstract::AbstractTuple>(maketuple_abs_inputs));
  // set abstract and attr
  auto matmul_dw_node_front = GetMatmulDwNodeFront(matmul_dw_nodes, backward_matmul_dx_dw_map);
  auto transpose_a1 = GetMatMulTransposeValue(matmul_dw_node_front, TRANSPOSE_A);
  auto transpose_b1 = GetMatMulTransposeValue(matmul_dw_node_front, TRANSPOSE_B);
  auto matmul_dw_node_front_input_node1_abstract = matmul_dw_node_front->input(1)->abstract();
  auto matmul_dw_node_front_input_node2_abstract = matmul_dw_node_front->input(kIndex2)->abstract();
  MS_EXCEPTION_IF_NULL(matmul_dw_node_front_input_node1_abstract);
  MS_EXCEPTION_IF_NULL(matmul_dw_node_front_input_node2_abstract);
  auto matmul_dw_node_front_input_node1_input_shape =
    matmul_dw_node_front_input_node1_abstract->BuildShape()->cast<abstract::ShapePtr>()->shape();
  auto matmul_dw_node_front_input_node2_input_shape =
    matmul_dw_node_front_input_node2_abstract->BuildShape()->cast<abstract::ShapePtr>()->shape();
  auto axis_pair = GetMatMulReduceAxis(matmul_dw_node_front_input_node2_input_shape.size(), transpose_a1, transpose_b1);
  auto axis1 = axis_pair.first;
  auto axis2 = axis_pair.second;
  auto axis1_value = NewValueNode(MakeValue<int64_t>(axis1));
  auto axis2_value = NewValueNode(MakeValue<int64_t>(axis2));
  UpdateValueNodeAbs(&axis1_value);
  UpdateValueNodeAbs(&axis2_value);
  matmul_dw_node_front_input_node1_input_shape[axis1] *= SizeToLong(matmul_dw_nodes.size());
  matmul_dw_node_front_input_node2_input_shape[axis2] *= SizeToLong(matmul_dw_nodes.size());
  auto concat1_shape_value = std::make_shared<abstract::Shape>(matmul_dw_node_front_input_node1_input_shape);
  auto concat2_shape_value = std::make_shared<abstract::Shape>(matmul_dw_node_front_input_node2_input_shape);
  CNodePtr concat;
  if (para_index == 1) {
    concat = graph->NewCNode({NewValueNode(prim::kPrimConcat->Clone()), maketuple, axis1_value});
    concat->set_abstract(matmul_dw_node_front_input_node1_abstract->Clone());
    concat->abstract()->set_shape(concat1_shape_value);
    auto concat_prim = GetCNodePrimitive(concat);
    concat_prim->set_attr(AXIS, MakeValue<int64_t>(axis1));
    concat_prim->set_attr(kAttrInputNums, MakeValue<int64_t>(maketuple_abs_inputs.size()));
    concat_prim->set_attr(kAttrConcatN, MakeValue<int64_t>(maketuple_abs_inputs.size()));
  } else {
    concat = graph->NewCNode({NewValueNode(prim::kPrimConcat->Clone()), maketuple, axis2_value});
    concat->set_abstract(matmul_dw_node_front_input_node2_abstract->Clone());
    concat->abstract()->set_shape(concat2_shape_value);
    auto concat_prim = GetCNodePrimitive(concat);
    concat_prim->set_attr(AXIS, MakeValue<int64_t>(axis2));
    concat_prim->set_attr(kAttrInputNums, MakeValue<int64_t>(maketuple_abs_inputs.size()));
    concat_prim->set_attr(kAttrConcatN, MakeValue<int64_t>(maketuple_abs_inputs.size()));
  }
  return concat;
}

void MergeMultiMatmulAssignAdd(const FuncGraphManagerPtr &manager, const FuncGraphPtr &each_graph,
                               const std::vector<CNodePtr> &matmul_dw_nodes,
                               const std::pair<AnfNodePtr, std::vector<AnfNodePtr>> &pair,
                               const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map,
                               const AnfNodePtr &forward_concat, size_t para_index) {
  auto concat1 = forward_concat;
  if (forward_concat == nullptr) {
    std::vector<AnfNodePtr> concat_inputs;
    (void)std::transform(matmul_dw_nodes.begin(), matmul_dw_nodes.end(), std::back_inserter(concat_inputs),
                         [](CNodePtr anf_node) { return anf_node->input(1); });
    concat1 = InsertConcat(matmul_dw_nodes, each_graph, 1, concat_inputs, backward_matmul_dx_dw_map);
    para_index = 1;
  }
  auto matmul_dw_node_front = GetMatmulDwNodeFront(matmul_dw_nodes, backward_matmul_dx_dw_map);
  std::vector<AnfNodePtr> concat_inputs;
  size_t para_index_sum = 3;
  (void)std::transform(
    matmul_dw_nodes.begin(), matmul_dw_nodes.end(), std::back_inserter(concat_inputs),
    [para_index, para_index_sum](CNodePtr anf_node) { return anf_node->input(para_index_sum - para_index); });
  auto concat =
    InsertConcat(matmul_dw_nodes, each_graph, para_index_sum - para_index, concat_inputs, backward_matmul_dx_dw_map);
  auto transpose_a1 = GetMatMulTransposeValue(matmul_dw_node_front, TRANSPOSE_A);
  auto transpose_b1 = GetMatMulTransposeValue(matmul_dw_node_front, TRANSPOSE_B);
  std::vector<AnfNodePtr> merged_matmul_inputs{NewValueNode(prim::kPrimMatMul), concat1, concat,
                                               NewValueNode(MakeValue(transpose_a1)),
                                               NewValueNode(MakeValue(transpose_b1))};
  if (para_index == kIndex2) {
    merged_matmul_inputs = {NewValueNode(prim::kPrimMatMul), concat, concat1, NewValueNode(MakeValue(transpose_a1)),
                            NewValueNode(MakeValue(transpose_b1))};
  }
  auto merged_matmul = each_graph->NewCNode(merged_matmul_inputs);
  merged_matmul->set_abstract(matmul_dw_node_front->abstract()->Clone());
  merged_matmul->input(kIndex3)->set_abstract(matmul_dw_node_front->input(kIndex3)->abstract()->Clone());
  merged_matmul->input(kIndex4)->set_abstract(matmul_dw_node_front->input(kIndex4)->abstract()->Clone());
  auto merged_matmul_prim = GetCNodePrimitive(merged_matmul);
  auto mat_prim = GetCNodePrimitive(matmul_dw_node_front);
  (void)merged_matmul_prim->SetAttrs(mat_prim->attrs());
  merged_matmul->set_attrs(matmul_dw_node_front->attrs());
  merged_matmul->set_primal_attrs(matmul_dw_node_front->primal_attrs());
  std::vector<std::string> unique_ids;
  for (const auto &dw_matmul : matmul_dw_nodes) {
    if (dw_matmul->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      (void)unique_ids.emplace_back(GetValue<std::string>(dw_matmul->GetPrimalAttr(kPrimalAttrForwardUniqueId)));
    }
  }
  merged_matmul->AddPrimalAttr(FORWARD_UNIQUE_ID_LIST, MakeValue<std::vector<std::string>>(unique_ids));
  auto replace_node = merged_matmul;
  // concat1 -> merged_matmul -> assign_add
  if (matmul_dw_node_front->HasAttr(kAttrCastDw)) {
    auto matmul_users = manager->node_users()[matmul_dw_node_front];
    CNodePtr cast_node = nullptr;
    for (const auto &user_pair : matmul_users) {
      if (IsPrimitiveCNode(user_pair.first, prim::kPrimCast) &&
          user_pair.first->cast<CNodePtr>()->HasAttr(kAttrCastDw)) {
        cast_node = user_pair.first->cast<CNodePtr>();
      }
    }
    if (!cast_node) {
      return;
    }
    std::vector<AnfNodePtr> cast_inputs{cast_node->input(0), merged_matmul, cast_node->input(kIndex2)};
    auto new_cast = each_graph->NewCNode(cast_inputs);
    new_cast->set_abstract(cast_node->abstract()->Clone());
    new_cast->abstract()->set_shape(merged_matmul->abstract()->GetShapeTrack());
    replace_node = new_cast;
  }

  std::vector<AnfNodePtr> assign_add_inputs{NewValueNode(prim::kPrimAssignAdd->Clone()), pair.first, replace_node};
  if (!pair.second.empty()) {
    auto assign_add_origin_cnode = dyn_cast<CNode>(pair.second[0]);
    MS_EXCEPTION_IF_NULL(assign_add_origin_cnode);
    auto final_input = assign_add_origin_cnode->input(assign_add_origin_cnode->size() - 1);
    if (HasAbstractUMonad(final_input)) {
      (void)assign_add_inputs.emplace_back(final_input);
    }
  }
  auto assign_add_cnode = each_graph->NewCNode(assign_add_inputs);
  for (const auto &assgin_add_origin_node : pair.second) {
    assign_add_cnode->set_abstract(assgin_add_origin_node->abstract()->Clone());
    manager->Replace(assgin_add_origin_node, assign_add_cnode);
  }
}

bool IsSameMatMul(const std::vector<CNodePtr> &matmul_dw_nodes) {
  auto matmul_dw_nodes_front = matmul_dw_nodes.front();
  return std::all_of(matmul_dw_nodes.begin(), matmul_dw_nodes.end(),
                     [&matmul_dw_nodes_front](const CNodePtr &matmul_dw_node) {
                       auto transpose_a1 = GetMatMulTransposeValue(matmul_dw_nodes_front, TRANSPOSE_A);
                       auto transpose_a2 = GetMatMulTransposeValue(matmul_dw_node, TRANSPOSE_A);
                       auto transpose_b1 = GetMatMulTransposeValue(matmul_dw_nodes_front, TRANSPOSE_B);
                       auto transpose_b2 = GetMatMulTransposeValue(matmul_dw_node, TRANSPOSE_B);
                       if (!(matmul_dw_nodes_front->HasAttr(kAttrCastDw) == matmul_dw_node->HasAttr(kAttrCastDw))) {
                         return false;
                       }

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
                       auto input_node_b1 = matmul_dw_nodes_front->input(kIndex2);
                       auto input_node_b2 = matmul_dw_node->input(kIndex2);
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
}

bool SkipAssignAddEliminate(const FuncGraphManagerPtr &manager,
                            const std::pair<AnfNodePtr, std::vector<AnfNodePtr>> &assign_add_map_pair,
                            std::vector<CNodePtr> *matmul_dw_nodes) {
  if (assign_add_map_pair.second.size() <= 1) {
    return true;
  }
  auto node_users = manager->node_users()[assign_add_map_pair.first];
  if (node_users.size() != assign_add_map_pair.second.size()) {
    return true;
  }
  // Check all input of assignadd node is matmul
  for (const auto &assign_add_node : assign_add_map_pair.second) {
    if (IsPrimitiveCNode(assign_add_node->cast<CNodePtr>()->input(kIndex2), prim::kPrimMatMul)) {
      auto matmul_node = assign_add_node->cast<CNodePtr>()->input(kIndex2)->cast<CNodePtr>();
      (void)matmul_dw_nodes->emplace_back(matmul_node);
    } else if (IsPrimitiveCNode(assign_add_node->cast<CNodePtr>()->input(kIndex2), prim::kPrimCast)) {
      auto cast_node = assign_add_node->cast<CNodePtr>()->input(kIndex2)->cast<CNodePtr>();
      if (IsPrimitiveCNode(cast_node->input(1), prim::kPrimMatMul)) {
        auto matmul_node = cast_node->input(1)->cast<CNodePtr>();
        (void)matmul_dw_nodes->emplace_back(matmul_node);
        matmul_node->AddAttr(kAttrCastDw, MakeValue(true));
        cast_node->AddAttr(kAttrCastDw, MakeValue(true));
      }
    } else {
      matmul_dw_nodes->clear();
      break;
    }
  }
  const size_t min_matmul_size = 2;
  return matmul_dw_nodes->size() < min_matmul_size;
}

bool SkipConcatEliminate(const std::vector<CNodePtr> &matmul_dw_nodes, const AnfNodePtr &partial, size_t *para_index,
                         std::vector<size_t> *bg_concat_input_index, std::vector<AnfNodePtr> *fg_concat_inputs) {
  if (matmul_dw_nodes.empty()) {
    return true;
  }
  auto bg = matmul_dw_nodes[0]->func_graph();
  auto manager = bg->manager();
  for (auto matmul_dw_node : matmul_dw_nodes) {
    for (size_t i = 0; i < bg->get_inputs().size(); i++) {
      if (bg->get_inputs()[i] == matmul_dw_node->input(1)) {
        auto user_param_size = manager->node_users()[bg->get_inputs()[i]].size();
        if (user_param_size != 1 || *para_index == kIndex2) {
          MS_LOG(INFO) << "Param is not only use by fusing concat,"
                       << "user node size of param is " << user_param_size;
          return true;
        }
        (void)bg_concat_input_index->emplace_back(i);
        if (i + kPartialPreIndex >= GetInputs(partial).size()) {
          return true;
        }
        (void)fg_concat_inputs->emplace_back(GetInputs(partial)[i + kPartialPreIndex]);
        *para_index = 1;
      } else if (bg->get_inputs()[i] == matmul_dw_node->input(kIndex2)) {
        auto user_param_size = manager->node_users()[bg->get_inputs()[i]].size();
        if (user_param_size != 1 || *para_index == 1) {
          MS_LOG(INFO) << "Param is not only use by fusing concat,"
                       << "user node size of param is " << user_param_size;
          return true;
        }
        (void)bg_concat_input_index->emplace_back(i);
        if (i + kPartialPreIndex >= GetInputs(partial).size()) {
          return true;
        }
        (void)fg_concat_inputs->emplace_back(GetInputs(partial)[i + kPartialPreIndex]);
        *para_index = kIndex2;
      }
    }
  }
  sort(bg_concat_input_index->begin(), bg_concat_input_index->end(), std::greater<size_t>());
  return bg_concat_input_index->size() != matmul_dw_nodes.size();
}

bool IsNeedOptimizeAssignAdd(bool is_kbyk_mode) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return false;
  }

  if (is_kbyk_mode && common::IsDisableRuntimeConfig(common::kRuntimeParalletAssignAddOpt)) {
    MS_LOG(INFO) << "KBK mode disable the assginadd opt.";
    return false;
  }
  return true;
}

void ReplaeAddNAssignAddToTwoAssignAdd(const FuncGraphManagerPtr &manager) {
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimAssignAdd)) {
        continue;
      }
      auto assign_add_input = GetInputNodeWithFilter(node->input(kIndex2), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad);
        return std::make_pair(filter, 1);
      });
      if (!IsPrimitiveCNode(assign_add_input, prim::kPrimAddN)) {
        continue;
      }
      auto addn_cnode = assign_add_input->cast<CNodePtr>();
      if (!IsPrimitiveCNode(addn_cnode->input(kIndex1), prim::kPrimMakeTuple)) {
        continue;
      }
      auto cast_node = GetInputNodeWithFilter(node->input(kIndex2), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimLoad);
        return std::make_pair(filter, 1);
      });
      bool contain_cast = IsPrimitiveCNode(cast_node, prim::kPrimCast);
      auto make_tuple_cnode = addn_cnode->input(kIndex1)->cast<CNodePtr>();
      std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple->Clone())};
      std::vector<AbstractBasePtr> maketuple_abs_inputs;
      auto matmuls = make_tuple_cnode->inputs();
      bool all_mm = std::all_of(matmuls.begin() + 1, matmuls.end(),
                                [&](auto cnode) { return IsPrimitiveCNode(cnode, prim::kPrimMatMul); });
      if (!all_mm) {
        continue;
      }
      for (size_t i = kIndex1; i < make_tuple_cnode->size(); ++i) {
        auto input_node = make_tuple_cnode->input(i);
        std::vector<AnfNodePtr> assign_add_inputs = node->inputs();
        assign_add_inputs[kIndex0] = NewValueNode(prim::kPrimAssignAdd->Clone());
        assign_add_inputs[kIndex2] = input_node;
        if (contain_cast) {
          std::vector<AnfNodePtr> cast_inputs{cast_node->cast<CNodePtr>()->input(kIndex0), input_node,
                                              cast_node->cast<CNodePtr>()->input(kIndex2)};
          auto new_cast = each_graph->NewCNode(cast_inputs);
          new_cast->set_abstract(cast_node->abstract()->Clone());
          new_cast->abstract()->set_shape(input_node->abstract()->GetShapeTrack());
          assign_add_inputs[kIndex2] = new_cast;
        }
        auto new_assign_add_cnode = each_graph->NewCNode(assign_add_inputs);
        new_assign_add_cnode->set_abstract(node->abstract()->Clone());
        make_tuple_inputs.push_back(new_assign_add_cnode);
        maketuple_abs_inputs.push_back(node->abstract()->Clone());
      }
      auto make_tuple = each_graph->NewCNode(make_tuple_inputs);
      make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(maketuple_abs_inputs));
      make_tuple->AddAttr(kAttrTmpMakeTuple, MakeValue(true));
      manager->Replace(node, make_tuple);
    }
  }
}

void EraseTmpMakeTuple(const FuncGraphManagerPtr &manager) {
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
        continue;
      }
      if (!node->HasAttr(kAttrTmpMakeTuple)) {
        continue;
      }
      auto node_inputs = node->inputs();
      if (node_inputs.size() < SIZE_TWO) {
        continue;
      }
      auto first_input = node_inputs[kIndex1];
      bool is_all_equal =
        std::all_of(node_inputs.begin() + 1, node_inputs.end(), [&](auto cnode) { return cnode == first_input; });
      if (!is_all_equal) {
        MS_LOG(INTERNAL_EXCEPTION) << "Merge assignAdd cannot erase make_tuple fail, make_tuple:"
                                   << node->DebugString();
      }
      manager->Replace(node, first_input);
    }
  }
}
}  // namespace

void AssignAddOpt(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!IsNeedOptimizeAssignAdd(ms_context->IsKByKExecutorMode())) {
    return;
  }
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ReplaeAddNAssignAddToTwoAssignAdd(manager);
  auto is_enable_concat_eliminate = ms_context->get_param<bool>(MS_CTX_ENABLE_CONCAT_ELIMINATE_OPT);
  MS_LOG(INFO) << "Merge multi matmul assign add begin and concat eliminate enable flag is:"
               << is_enable_concat_eliminate;
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> assign_add_map;
    std::unordered_map<CNodePtr, CNodePtr> backward_matmul_dx_dw_map;
    ExtractBackwardMatMul(origin_nodes_topological, &backward_matmul_dx_dw_map);
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimAssignAdd)) {
        continue;
      }
      (void)assign_add_map[node->input(1)].emplace_back(node);
    }
    for (const auto &pair : assign_add_map) {
      std::vector<CNodePtr> matmul_dw_nodes;
      if (SkipAssignAddEliminate(manager, pair, &matmul_dw_nodes)) {
        continue;
      }
      if (!IsSameMatMul(matmul_dw_nodes)) {
        return;
      }
      // 1. fine forward graph
      FuncGraphPtr fg;
      CNodePtr cnode;
      for (auto &entry : each_graph->func_graph_cnodes_index()) {
        cnode = entry.first->first->cast<CNodePtr>();
        auto index = entry.first->second;
        if (cnode->inputs().size() >= kIndex2 && index == 1 && IsPrimitive(cnode->inputs().at(0), prim::kPrimPartial)) {
          // To find real calling.
          fg = cnode->func_graph();
          MS_EXCEPTION_IF_NULL(fg);
        } else {
          return;
        }
      }
      // 2. insert concat in forward
      // 2.1 find concat index of backward graph
      std::vector<size_t> bg_concat_input_index;
      std::vector<AnfNodePtr> fg_concat_inputs;
      size_t para_index = 0;
      if (SkipConcatEliminate(matmul_dw_nodes, cnode, &para_index, &bg_concat_input_index, &fg_concat_inputs) ||
          !is_enable_concat_eliminate) {
        // input of matmul is not param.
        MergeMultiMatmulAssignAdd(manager, each_graph, matmul_dw_nodes, pair, backward_matmul_dx_dw_map, nullptr,
                                  para_index);
        continue;
      }
      // 2.2 insert concat1 in forward
      auto concat = InsertConcat(matmul_dw_nodes, fg, para_index, fg_concat_inputs, backward_matmul_dx_dw_map);
      // 3. remove output edges from forward graph and input edge from backward graph
      auto partial_cnode_inputs(cnode->inputs());
      auto new_backward_parameters(each_graph->parameters());

      for (size_t i = 0; i < bg_concat_input_index.size(); i++) {
        if (bg_concat_input_index[i] + kPartialPreIndex >= partial_cnode_inputs.size() ||
            bg_concat_input_index[i] > new_backward_parameters.size()) {
          MS_LOG_EXCEPTION << "Erase index out of partial_inputs size";
        }
        (void)partial_cnode_inputs.erase(partial_cnode_inputs.begin() + bg_concat_input_index[i] + kPartialPreIndex);
        (void)new_backward_parameters.erase(new_backward_parameters.begin() + bg_concat_input_index[i]);
      }
      if (bg_concat_input_index.back() + kPartialPreIndex > partial_cnode_inputs.size() ||
          bg_concat_input_index.back() > new_backward_parameters.size()) {
        MS_LOG_EXCEPTION << "Insert index out of partial_inputs size";
      }
      (void)partial_cnode_inputs.insert(partial_cnode_inputs.begin() + bg_concat_input_index.back() + kPartialPreIndex,
                                        concat);
      auto new_partial_cnode = fg->NewCNode(partial_cnode_inputs);
      (void)manager->Replace(cnode, new_partial_cnode);
      auto new_parameter = each_graph->add_parameter();
      new_parameter->set_abstract(concat->abstract()->Clone());
      (void)new_backward_parameters.insert(new_backward_parameters.begin() + bg_concat_input_index.back(),
                                           new_parameter);
      each_graph->set_parameters(new_backward_parameters);
      // 4. insert concat2 in backward graph
      MergeMultiMatmulAssignAdd(manager, each_graph, matmul_dw_nodes, pair, backward_matmul_dx_dw_map, new_parameter,
                                para_index);
    }
  }
  EraseTmpMakeTuple(manager);
}
}  // namespace parallel
}  // namespace mindspore
