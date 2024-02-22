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

#include "frontend/parallel/step_assigned_parallel.h"

#include <cinttypes>
#include <ctime>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/ops_info/tmp_identity_info.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
// l_RefMap, for CNode B input i is a RefKey[Parameter C],
// it will be one item in map with key: C, and value: (B, i)
std::map<AnfNodePtr, std::pair<AnfNodePtr, int64_t>> l_RefMap;

static std::shared_ptr<TensorLayout> GetOutputLayoutFromCNode(const CNodePtr &cnode, size_t output_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  if (distribute_operator->outputs_tensor_info().size() <= output_index) {
    MS_LOG(EXCEPTION) << "outputs_tensor_info size is  " << distribute_operator->inputs_tensor_info().size()
                      << ", must be greater than output_index  " << output_index;
  }
  TensorInfo tensorinfo_out = distribute_operator->outputs_tensor_info()[output_index];
  TensorLayout tensorlayout_out = tensorinfo_out.tensor_layout();
  return std::make_shared<TensorLayout>(tensorlayout_out);
}

static std::shared_ptr<TensorLayout> FindPrevParallelCareNodeLayout(const AnfNodePtr &node, size_t output_index) {
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, output_index);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  return nullptr;
}

static std::shared_ptr<TensorLayout> FindPrevLayout(const AnfNodePtr &node) {
  if (node->isa<Parameter>()) {
    return CreateParameterLayout(node);
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return cnode->user_data<TensorLayout>();
  }
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>() &&
      !IsPrimitiveCNode(node, prim::kPrimReshape)) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, 0);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  if (prim->name() == prim::kPrimTupleGetItem->name()) {
    auto tuple_index = GetTupleGetItemIndex(cnode);
    auto layout_ptr = FindPrevParallelCareNodeLayout(cnode->input(1), LongToSize(tuple_index));
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << " Failure:FindPrevLayout failed, tuple_getitem before reshape, but there does not exit a "
                           "parallel care node "
                           "before tuple_getitem!";
    }
    return layout_ptr;
  }
  for (size_t index = 0; index < cnode->size(); ++index) {
    if (prim->name() == DEPEND && index != 1) {
      continue;
    }
    auto layout_ptr = FindPrevLayout(cnode->inputs()[index]);
    if (!layout_ptr) {
      continue;
    }
    return layout_ptr;
  }
  MS_LOG(WARNING) << "FindPrevLayout return nullptr, if reshape is not the first primitive, there must be some error";
  return nullptr;
}

// if reshape's output connect to several primitive, return the first layout found
static std::shared_ptr<TensorLayout> FindNextLayout(const CNodePtr &cnode, bool *next_is_reshape,
                                                    int make_tuple_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[cnode];
  for (auto &node_pair : node_set) {
    auto use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    if (IsPrimitiveCNode(use_apply, prim::kPrimReshape)) {
      *next_is_reshape = true;
      continue;
    }
    if (IsPrimitiveCNode(use_apply, prim::kPrimDepend) && node_pair.second != 1) {
      continue;
    }
    if (IsPrimitiveCNode(use_apply, prim::kPrimMakeTuple)) {
      make_tuple_index = node_pair.second;
      return FindNextLayout(use_apply, next_is_reshape, make_tuple_index);
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      if (make_tuple_index != -1) {
        node_pair.second = make_tuple_index;
      }
      MS_LOG(INFO) << "FindNextLayout success node " << use_apply->DebugString();
      *next_is_reshape = false;
      auto layout = GetInputLayoutFromCNode(node_pair);
      return std::make_shared<TensorLayout>(layout);
    }
    MS_LOG(DEBUG) << "FindNextLayout failed node " << use_apply->DebugString() << "  " << IsParallelCareNode(use_apply)
                  << "   " << use_apply->has_user_data<OperatorInfo>();

    auto layout_ptr = FindNextLayout(use_apply, next_is_reshape, -1);
    if (layout_ptr) {
      return layout_ptr;
    }
  }
  MS_LOG(WARNING) << "FindNextLayout return nullptr, if reshape is not the last primitive, there must be some error";
  return nullptr;
}

AnfNodePtr NewAllGatherNode(const std::string &name, const std::string &group) {
  std::shared_ptr<Primitive> prim;
  prim = std::make_shared<Primitive>(name);
  ValuePtr attr0_value = MakeValue(group);
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  prim->AddAttr(GROUP, attr0_value);
  prim->AddAttr("fusion", MakeValue(static_cast<int64_t>(0)));
  prim->AddAttr("mean_flag", MakeValue(false));
  prim->AddAttr("no_eliminate", MakeValue(true));
  std::vector<unsigned int> rank_list = {};
  auto long_rank_list = parallel::g_device_manager->FindRankListByHashName(group);
  (void)std::transform(long_rank_list.begin(), long_rank_list.end(), std::back_inserter(rank_list),
                       [](int64_t d) -> unsigned int { return IntToUint(LongToInt(d)); });

  prim->AddAttr(kAttrRankSize, MakeValue(static_cast<int64_t>(rank_list.size())));
  auto node = NewValueNode(prim);
  return node;
}

// From ops To AllReduce->ops
static void InsertAllReduceToNodeInput(const CNodePtr &node, const std::string &group,
                                       const std::string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  size_t index = 1;
  MS_EXCEPTION_IF_NULL(func_graph);
  Operator allreduce_op = CreateAllReduceOp(REDUCE_OP_SUM, group);

  // Insert it as the input of the node
  AnfNodePtr input = node->input(index);
  MS_EXCEPTION_IF_NULL(input);
  // if it is not a tensor, continue
  if ((!input->isa<CNode>() && !input->isa<Parameter>()) || HasAbstractMonad(input)) {
    return;
  }
  InsertNode(allreduce_op, node, index, node->input(index), func_graph, instance_name);
}

bool InsertAllReduceOps(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, const size_t devices) {
  int64_t device_num = devices;
  if (device_num <= 1) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto expect_add = node->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_add, prim::kPrimAdd->name())) {
      continue;
    }
    AnfNodePtr expect_matmul = expect_add->input(1);
    MS_EXCEPTION_IF_NULL(expect_matmul);
    if (!expect_matmul->isa<CNode>()) {
      continue;
    }
    auto expect_matmul_cnode = expect_matmul->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_matmul_cnode, prim::kPrimMatMul->name())) {
      continue;
    }
    auto matmul_prim = GetCNodePrimitive(expect_matmul_cnode);
    MS_EXCEPTION_IF_NULL(matmul_prim);
    if (matmul_prim->HasAttr(IN_STRATEGY)) {
      auto matmul_stra = matmul_prim->GetAttr(IN_STRATEGY);
      if (matmul_stra == nullptr) {
        continue;
      }
      auto matmul_var = GetValue<vector<Shape>>(matmul_stra);
      if (matmul_var.size() > 0) {
        Dimensions sub_a_strategy = matmul_var.at(0);
        Dimensions sub_b_strategy = matmul_var.at(1);
        if (sub_a_strategy.size() == 2 && sub_b_strategy.size() == 2 && sub_a_strategy[1] == sub_b_strategy[0] &&
            sub_a_strategy[1] > 1) {
          MS_LOG(INFO) << "Here should insert AllReduce Ops: ";
          InsertAllReduceToNodeInput(expect_add, HCCL_WORLD_GROUP, PARALLEL_GLOBALNORM);
          AnfNodePtr expect_reshape = expect_matmul_cnode->input(1);
          if (!expect_reshape->isa<CNode>()) {
            continue;
          }
          auto expect_reshape_cnode = expect_reshape->cast<CNodePtr>();
          if (!IsSomePrimitive(expect_reshape_cnode, prim::kPrimReshape->name())) {
            continue;
          }
          Shape origin_dst_shape =
            GetValue<std::vector<int64_t>>(expect_reshape_cnode->input(2)->cast<ValueNodePtr>()->value());
          if (origin_dst_shape.size() == 1 && origin_dst_shape[0] == -1) {
            continue;
          }
          Shape new_dst_shape;
          new_dst_shape.push_back(origin_dst_shape[0]);
          new_dst_shape.push_back(origin_dst_shape[1] / device_num);
          for (auto s : new_dst_shape) {
            MS_LOG(INFO) << "new_dst_shape: " << s;
          }

          expect_reshape_cnode->set_input(2, NewValueNode(MakeValue(new_dst_shape)));

          auto reshape_node_abstract = expect_reshape_cnode->abstract()->Clone();
          std::shared_ptr<abstract::BaseShape> output_shape = std::make_shared<abstract::Shape>(new_dst_shape);
          reshape_node_abstract->set_shape(output_shape);
          MS_LOG(INFO) << "new_dst_shape: " << reshape_node_abstract->ToString();
          expect_reshape_cnode->set_abstract(reshape_node_abstract);
        }
      }
    }
  }
  return true;
}

bool InsertAllReduceOpsForFFN(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root,
                              const size_t devices) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto expect_add = node->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_add, prim::kPrimAdd->name())) {
      continue;
    }
    AnfNodePtr expect_batchmatmul = expect_add->input(1);
    MS_EXCEPTION_IF_NULL(expect_batchmatmul);
    if (!expect_batchmatmul->isa<CNode>()) {
      continue;
    }
    auto expect_batchmatmul_cnode = expect_batchmatmul->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_batchmatmul_cnode, prim::kPrimBatchMatMul->name())) {
      continue;
    }
    auto batchmatmul_prim = GetCNodePrimitive(expect_batchmatmul_cnode);
    MS_EXCEPTION_IF_NULL(batchmatmul_prim);
    if (batchmatmul_prim->HasAttr(IN_STRATEGY)) {
      auto batchmatmul_stra = batchmatmul_prim->GetAttr(IN_STRATEGY);
      if (batchmatmul_stra == nullptr) {
        continue;
      }
      auto batchmatmul_var = GetValue<vector<Shape>>(batchmatmul_stra);
      if (batchmatmul_var.size() > 0) {
        Dimensions sub_a_strategy = batchmatmul_var.at(0);
        Dimensions sub_b_strategy = batchmatmul_var.at(1);
        if (sub_a_strategy.size() == 4 && sub_b_strategy.size() == 3 && sub_a_strategy[3] == sub_b_strategy[1] &&
            sub_a_strategy[3] > 1) {
          MS_LOG(INFO) << "Here should insert AllReduce Ops: ";
          InsertAllReduceToNodeInput(expect_add, HCCL_WORLD_GROUP, PARALLEL_GLOBALNORM);
        }
      }
    }
  }
  return true;
}

void ChangeReshape(const AnfNodePtr &node, const size_t devices) {
  int64_t device_num = devices;
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return;
  }
  auto expect_reshape_cnode = node->cast<CNodePtr>();
  if (!IsSomePrimitive(expect_reshape_cnode, prim::kPrimReshape->name())) {
    return;
  }
  auto reshape_node_input = expect_reshape_cnode->input(2);
  if (reshape_node_input == nullptr) {
    return;
  }
  MS_LOG(INFO) << "find reshape ops: " << expect_reshape_cnode->DebugString();
  if (reshape_node_input->isa<ValueNode>()) {
    Shape origin_dst_shape = GetValue<std::vector<int64_t>>(reshape_node_input->cast<ValueNodePtr>()->value());
    if (origin_dst_shape.size() != 4) {
      return;
    }
    if (origin_dst_shape[2] % device_num != 0) {
      return;
    }
    Shape new_dst_shape;
    new_dst_shape.push_back(origin_dst_shape[0]);
    new_dst_shape.push_back(origin_dst_shape[1]);
    new_dst_shape.push_back(origin_dst_shape[2] / device_num);
    new_dst_shape.push_back(origin_dst_shape[3]);
    for (auto s : new_dst_shape) {
      MS_LOG(INFO) << "reshape new_dst_shape: " << s;
    }
    expect_reshape_cnode->set_input(2, NewValueNode(MakeValue(new_dst_shape)));
    auto reshape_node_abstract = expect_reshape_cnode->abstract()->Clone();
    std::shared_ptr<abstract::BaseShape> output_shape = std::make_shared<abstract::Shape>(new_dst_shape);
    reshape_node_abstract->set_shape(output_shape);
    MS_LOG(INFO) << "new_dst_shape: " << reshape_node_abstract->ToString();
    expect_reshape_cnode->set_abstract(reshape_node_abstract);

  } else if (reshape_node_input->isa<CNode>()) {
    auto expect_maketuple_cnode = reshape_node_input->cast<CNodePtr>();
    MS_LOG(INFO) << "Before modify reshape maketuple: " << expect_maketuple_cnode->DebugString();
    if (!IsSomePrimitive(expect_maketuple_cnode, prim::kPrimMakeTuple->name())) {
      return;
    }
    auto maketuple_node_input = expect_maketuple_cnode->input(3);
    if (maketuple_node_input == nullptr) {
      return;
    }
    if (!maketuple_node_input->isa<ValueNode>()) {
      return;
    }
    int64_t origin_value = GetValue<int64_t>(maketuple_node_input->cast<ValueNodePtr>()->value());
    if (origin_value % device_num == 0 && !expect_maketuple_cnode->HasAttr("has_modifyed")) {
      int64_t new_value = origin_value / device_num;
      expect_maketuple_cnode->set_input(3, NewValueNode(MakeValue(new_value)));
      expect_maketuple_cnode->AddAttr("has_modifyed", MakeValue(true));
      MS_LOG(INFO) << "After modify reshape maketuple: " << expect_maketuple_cnode->DebugString();
    }
  }
}

bool ModifyReshapeOps(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, const size_t devices) {
  int64_t device_num = devices;
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto expect_transpose = node->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_transpose, prim::kPrimTranspose->name())) {
      continue;
    }
    auto transpose_prim = GetCNodePrimitive(expect_transpose);
    MS_EXCEPTION_IF_NULL(transpose_prim);
    if (!transpose_prim->HasAttr(IN_STRATEGY)) {
      continue;
    }
    auto transpose_stra = transpose_prim->GetAttr(IN_STRATEGY);
    if (transpose_stra == nullptr) {
      continue;
    }
    auto transpose_var = GetValue<vector<Shape>>(transpose_stra);
    if (transpose_var.size() > 0) {
      Dimensions sub_strategy = transpose_var.at(0);
      bool all_ones = std::all_of(sub_strategy.begin(), sub_strategy.end(), [](int64_t i) { return i == 1; });
      if (all_ones) {
        continue;
      }
    }
    AnfNodePtr expect_reshape = expect_transpose->input(1);
    ChangeReshape(expect_reshape, device_num);
  }
  return true;
}

bool ModifyMakeTupleOps(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, const size_t devices) {
  int64_t device_num = devices;
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto expect_maketuple = node->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_maketuple, prim::kPrimMakeTuple->name())) {
      continue;
    }
    if (expect_maketuple->size() != 4) {
      continue;
    }
    if (expect_maketuple->input(1)->isa<CNode>() && expect_maketuple->input(2)->isa<CNode>() &&
        expect_maketuple->input(3)->isa<ValueNode>()) {
      if (IsSomePrimitive(expect_maketuple->input(1)->cast<CNodePtr>(), prim::kPrimTupleGetItem->name()) &&
          IsSomePrimitive(expect_maketuple->input(2)->cast<CNodePtr>(), prim::kPrimTupleGetItem->name())) {
        auto maketuple_node_input = expect_maketuple->input(3);
        int64_t origin_value = GetValue<int64_t>(maketuple_node_input->cast<ValueNodePtr>()->value());
        if (origin_value % device_num == 0) {
          int64_t new_value = origin_value / device_num;
          expect_maketuple->set_input(3, NewValueNode(MakeValue(new_value)));
          MS_LOG(INFO) << "After modify MakeTuple, the shape is : " << expect_maketuple->DebugString();
        }
      }
    }
  }
  return true;
}

bool ModifySoftmaxReshapeOps(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, const size_t devices) {
  int64_t device_num = devices;
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto expect_reshape = node->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_reshape, prim::kPrimReshape->name())) {
      continue;
    }

    AnfNodePtr expect_cast = expect_reshape->input(1);
    MS_EXCEPTION_IF_NULL(expect_cast);
    if (!expect_cast->isa<CNode>()) {
      continue;
    }
    auto expect_cast_cnode = expect_cast->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_cast_cnode, "Cast")) {
      continue;
    }

    auto expect_softmax = expect_cast_cnode->input(1);
    MS_EXCEPTION_IF_NULL(expect_softmax);
    if (!expect_softmax->isa<CNode>()) {
      continue;
    }
    auto expect_softmax_cnode = expect_softmax->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_softmax_cnode, "Softmax")) {
      continue;
    }
    auto reshape_node_input = expect_reshape->input(2);
    if (reshape_node_input == nullptr) {
      continue;
    }
    if (!reshape_node_input->isa<ValueNode>()) {
      continue;
    }
    Shape origin_dst_shape = GetValue<std::vector<int64_t>>(reshape_node_input->cast<ValueNodePtr>()->value());
    if (origin_dst_shape.size() != 4) {
      continue;
    }
    if (origin_dst_shape[1] % device_num != 0) {
      continue;
    }
    Shape new_dst_shape;
    new_dst_shape.push_back(origin_dst_shape[0]);
    new_dst_shape.push_back(origin_dst_shape[1] / device_num);
    new_dst_shape.push_back(origin_dst_shape[2]);
    new_dst_shape.push_back(origin_dst_shape[3]);
    for (auto s : new_dst_shape) {
      MS_LOG(INFO) << "reshape new_dst_shape: " << s;
    }

    expect_reshape->set_input(2, NewValueNode(MakeValue(new_dst_shape)));

    auto reshape_node_abstract = expect_reshape->abstract()->Clone();
    std::shared_ptr<abstract::BaseShape> output_shape = std::make_shared<abstract::Shape>(new_dst_shape);
    reshape_node_abstract->set_shape(output_shape);
    MS_LOG(INFO) << "new_dst_shape: " << reshape_node_abstract->ToString();
    expect_reshape->set_abstract(reshape_node_abstract);
  }
  return true;
}

static bool CheckExtractInformation(const CNodePtr &cnode) {
  if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }

  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  if ((prim->name() == MAKE_TUPLE) || (prim->name() == MAKE_LIST) || (prim->name() == RECEIVE)) {
    return false;
  }
  if (!IsParallelCareNode(cnode)) {
    return false;
  }
  return true;
}

void InitRefMap(const FuncGraphPtr &root) {
  auto manager = root->manager();
  auto node_list = TopoSort(root->get_return());
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    if ((prim->name() == MAKE_TUPLE) || (prim->name() == MAKE_LIST) || (prim->name() == RECEIVE)) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimSend) || IsPrimitiveCNode(node, prim::kPrimUpdateState) ||
        IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }
    std::vector<AnfNodePtr> all_inputs = cnode->inputs();
    size_t inputs_size = all_inputs.size();
    for (size_t i = 1; i < inputs_size; ++i) {
      AnfNodePtr input = all_inputs[i];
      if (HasAbstractMonad(input)) {
        continue;
      }
      if (input->isa<Parameter>() && input->cast<ParameterPtr>()->has_default()) {
        auto func_graph = cnode->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        auto param_node = input->cast<ParameterPtr>();
        std::pair<AnfNodePtr, int64_t> node_pair = std::make_pair(cnode, SizeToLong(i));
        if (IsInTrivialNodeList(cnode) || IsSomePrimitive(cnode, prim::kPrimLoad->name())) {
          auto &node_users = manager->node_users();
          auto iter = node_users.find(node);
          if (iter == node_users.end()) {
            MS_LOG(ERROR) << "Can not find the parameter used node.";
          }
          auto &node_set = iter->second;
          const auto node_set_back = node_set.back().first->cast<CNodePtr>();
          if (node_set_back != nullptr && IsSomePrimitive(node_set_back, prim::kPrimMakeTuple->name())) {
            l_RefMap[param_node] = node_set.front();
          } else {
            l_RefMap[param_node] = node_set.back();
          }
        } else {
          l_RefMap[param_node] = node_pair;
        }
      }
    }
  }
}

static void SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int64_t> &res, size_t rank_id) {
  MS_LOG(INFO) << "Begin set parallel shape";
  // check null for param and cnode
  auto param_shape = parameter->Shape();

  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(param_shape);

  CNodePtr cnode = res.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = cnode->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "node " << cnode->DebugString() << " 's distribute_operator is nullptr";
  }
  if (LongToSize(res.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The parameter index is not in inputs_tensor_info. index = " << (res.second - 1)
                      << ", inputs_tensor_info size = " << distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(res.second - 1)];
  TensorLayout tensor_layout = tensorinfo_in.tensor_layout();
  Shape slice_shape = tensor_layout.slice_shape().array();

  AbstractBasePtr abstract = parameter->abstract();
  if (abstract == nullptr) {
    MS_LOG(EXCEPTION) << "parameter " << parameter->ToString() << ": abstract is nullptr";
  }

  AbstractBasePtr cloned_abstract = abstract->Clone();
  if (cloned_abstract == nullptr) {
    MS_LOG(EXCEPTION) << "parameter " << parameter->ToString() << ": abstract clone failed";
  }

  cloned_abstract->set_shape(std::make_shared<abstract::Shape>(slice_shape));
  parameter->set_abstract(cloned_abstract);
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();

  MS_EXCEPTION_IF_NULL(parameter_ptr);
  MS_LOG(INFO) << "Begin split parameters";
  parameter_ptr->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_layout));
  if (ParallelContext::GetInstance()->direct_split() && parameter_ptr->has_default()) {
    auto layout = parameter_ptr->user_data<TensorLayout>();
    MS_LOG(INFO) << "parameter: " << parameter->ToString() << parameter->Shape()->ToString()
                 << "parameter_ptr->default_param()" << parameter_ptr->default_param() << "LAYOUT"
                 << layout->ToString();
    SliceTensorObj(parameter_ptr, layout, rank_id);
  }
}

static void DoParameterSliceShape(const FuncGraphPtr &root, size_t rank_id) {
  MS_EXCEPTION_IF_NULL(root);
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter->Shape());
    auto iter = l_RefMap.find(parameter);
    if (iter != l_RefMap.cend()) {
      MS_LOG(INFO) << "SetParallelShape for parameter: " << parameter->ToString();
      SetParallelShape(parameter, l_RefMap[parameter], rank_id);
      SetSharedParameterFlag(root, parameter);
      continue;
    }
  }
  l_RefMap.clear();
}

StrategyPtr ExtractAndModifyStrategy(const CNodePtr &cnode, const std::string &attr_name, const ValuePtr &stra) {
  if (stra == nullptr) {
    return nullptr;
  }
  auto var = stra->cast<ValueTuplePtr>();
  if (var == nullptr) {
    return nullptr;
  }

  StrategyPtr strategyPtr;
  int64_t stage_id = g_device_manager->stage_id();
  MS_LOG(INFO) << "Extract information: strategy " << stra->ToString();
  int64_t device_num = g_device_manager->DeviceNum();
  MS_LOG(INFO) << "Extract information: device_num " << device_num;
  if (var->size() > 0) {
    std::vector<ValuePtr> elements = var->value();
    Strategies strategy;
    for (uint64_t index = 0; index < elements.size(); ++index) {
      Dimensions dim;
      if (elements[index]->isa<ValueSequence>()) {
        auto value_tuple = elements[index]->cast<ValueTuplePtr>();
        std::vector<ValuePtr> value_vector = value_tuple->value();
        (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                             [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
        for (size_t i = 0; i < dim.size(); i++) {
          if (dim[i] > 1 && dim[i] != device_num) {
            dim[i] = device_num;
          }
        }
        strategy.push_back(dim);
      } else {
        MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
      }
    }
    if (strategy.empty()) {
      MS_LOG(EXCEPTION) << "ExtractStrategy: failed to extract strategy";
    }
    cnode->AddPrimalAttr(attr_name, MakeValue(strategy));
    strategyPtr = NewStrategy(stage_id, strategy);
    MS_LOG(INFO) << "Extract information: new strategy " << cnode->GetPrimalAttr(attr_name)->ToString();
  }
  return strategyPtr;
}

static void ExtractStrategyAndInit(const CNodePtr &cnode, const PrimitivePtr &prim, const OperatorInfoPtr &op_info) {
  StrategyPtr in_strategy = nullptr, out_strategy = nullptr;
  auto attrs = prim->attrs();

  // load strategy map from checkpoint
  StrategyMap stra_map;

  std::string strategy_key_name = "";
  auto param_names = NodeParameterName(cnode, -1, 0);
  if (!param_names.empty()) {
    strategy_key_name = prim->name() + "_" + param_names[0].first;
  }
  if (!prim->HasAttr(STAND_ALONE)) {
    if ((!StrategyFound(attrs) && !cnode->HasPrimalAttr(IN_STRATEGY)) || prim->HasAttr(BATCH_PARALLEL)) {
      MS_LOG(INFO) << "ExtractInformation: the strategy of node " << cnode->ToString() << " prim " << prim->name()
                   << " is empty, using batch parallel";
      in_strategy = GenerateBatchParallelStrategy(op_info, prim);
    } else if (cnode->HasPrimalAttr(IN_STRATEGY)) {
      in_strategy = ExtractAndModifyStrategy(cnode, IN_STRATEGY, cnode->GetPrimalAttr(IN_STRATEGY));

      out_strategy = ExtractAndModifyStrategy(cnode, OUT_STRATEGY, cnode->GetPrimalAttr(OUT_STRATEGY));
    } else if (StrategyFound(attrs)) {
      in_strategy = ExtractAndModifyStrategy(cnode, IN_STRATEGY, attrs[IN_STRATEGY]);
      out_strategy = ExtractAndModifyStrategy(cnode, OUT_STRATEGY, attrs[OUT_STRATEGY]);
    } else {
      in_strategy = stra_map[strategy_key_name];
    }
  } else {
    in_strategy = GenerateStandAloneStrategy(op_info->inputs_shape());
  }

  MS_EXCEPTION_IF_NULL(in_strategy);
  if (op_info->Init(in_strategy, out_strategy) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " init failed" << trace::DumpSourceLines(cnode);
  }
}

void ExtractGraphInformation(const std::vector<AnfNodePtr> &all_nodes) {
  MS_LOG(INFO) << "ExtractInformation";
  SetStridedSliceSplitStrategy(all_nodes);
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!CheckExtractInformation(cnode) || IsPrimitiveCNode(node, prim::kPrimSend) ||
        IsPrimitiveCNode(node, std::make_shared<Primitive>("PadV3")) ||
        IsPrimitiveCNode(node, std::make_shared<Primitive>("StridedSlice")) ||
        IsPrimitiveCNode(node, std::make_shared<Primitive>("Sort")) ||
        IsPrimitiveCNode(node, std::make_shared<Primitive>("Less")) ||
        IsPrimitiveCNode(node, std::make_shared<Primitive>("Range"))) {
      continue;
    }

    SetVirtualDatasetStrategy(cnode);
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);

    OperatorInfoPtr operator_ = CreateOperatorInfo(cnode);
    operator_->set_assigned_parallel(true);
    MS_EXCEPTION_IF_NULL(operator_);

    if (prim->name() == RESHAPE) {
      cnode->set_user_data<OperatorInfo>(operator_);
      continue;
    }

    ExtractStrategyAndInit(cnode, prim, operator_);
    cnode->set_user_data<OperatorInfo>(operator_);
  }
}

static void StepReplaceGraph(const ReplaceGraphPtr &replace_graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(replace_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(replace_graph->second);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }
  mindspore::HashMap<AnfNodePtr, int> input_map = {};
  static int appear_count = 0;
  for (auto &replace_input : replace_graph->first) {
    auto pre_node = node->input(LongToSize(replace_input.second));

    auto it = input_map.find(replace_input.first);
    if (it != input_map.end()) {
      appear_count = 1 + it->second;
    } else {
      appear_count = 1;
    }
    auto replace_input_cnode = replace_input.first->cast<CNodePtr>();
    size_t inputs_size = replace_input_cnode->size();
    while (IntToSize(appear_count) < inputs_size && replace_input_cnode->input(appear_count)->func_graph() != nullptr) {
      ++appear_count;
    }
    if (IntToSize(appear_count) >= inputs_size) {
      MS_LOG(EXCEPTION) << "No replaceable virtual_input_node";
    }
    input_map[replace_input.first] = appear_count;
    replace_input_cnode->set_in_forward_flag(true);
    manager->SetEdge(replace_input.first, appear_count, pre_node);
  }
  //  "(void)manager->Replace(replace_graph->first, pre_node);" can not be called
  auto replace_output = replace_graph->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(replace_output);
  replace_output->set_in_forward_flag(true);
  replace_output->set_primal_attrs(node->primal_attrs());
  (void)manager->Replace(node, replace_output);
}

static void ReplaceGatherOps(const std::vector<AnfNodePtr> &all_nodes, const size_t devices) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (!IsSomePrimitive(cnode, prim::kPrimGather->name())) {
        continue;
      }
      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      MS_EXCEPTION_IF_NULL(distribute_operator);
      auto replace_op = distribute_operator->replace_op();
      // StepReplaceGraph: after calling StepReplaceGraph, cnode can not be used anymore.
      auto replace_graph = distribute_operator->replace_graph(cnode);
      if (!replace_op.empty() && replace_graph) {
        MS_LOG(EXCEPTION) << "Only one of replace_op or replace_op can be used";
      }
      if (replace_graph) {
        MS_LOG(INFO) << "StepReplaceGraph " << cnode->DebugString();
        StepReplaceGraph(replace_graph, cnode);
      }
    }
  }
}

static void FixReturnRedistribution(const FuncGraphPtr &root, const size_t devices) {
  MS_LOG(INFO) << "FixReturnRedistribution";
  CNodePtr ret = root->get_return();
  AnfNodePtr expect_matmul = ret->input(1);
  MS_EXCEPTION_IF_NULL(expect_matmul);
  if (!expect_matmul->isa<CNode>()) {
    return;
  }
  auto expect_matmul_node = expect_matmul->cast<CNodePtr>();
  if (!IsSomePrimitive(expect_matmul_node, prim::kPrimMatMul->name())) {
    return;
  }
  Shapes return_input_shapes = GetNodeShape(ret);
  MS_LOG(INFO) << "return_input_shapes size" << return_input_shapes.size();
  if (return_input_shapes.size() == 1) {
    MS_LOG(INFO) << "return_input_shapes: " << return_input_shapes[0][0] << return_input_shapes[0][1];
    GenerateGraph gen_g = GenerateGraph(expect_matmul->cast<CNodePtr>()->attrs());
    if (gen_g.Init(ret) != SUCCESS) {
      MS_LOG(ERROR) << "MatMul->Return"
                    << "GenerateGraph Init failed";
    }

    Attr transpose_a_attr = std::make_pair(TRANSPOSE_A, MakeValue(false));
    Attr transpose_b_attr = std::make_pair(TRANSPOSE_B, MakeValue(true));
    OperatorAttrs matmul_attrs = {transpose_a_attr, transpose_b_attr};
    auto matmul = gen_g.PushBack({gen_g.NewOpInst(prim::kPrimMatMul->name(), matmul_attrs), gen_g.virtual_input_node(),
                                  gen_g.virtual_input_node()});

    if (return_input_shapes[0][0] == 1) {
      auto des_shape = return_input_shapes[0];
      auto des_size = return_input_shapes[0][1];
      auto origin_size = des_size / devices;
      Shape origin_shape;
      origin_shape.push_back(origin_size);
      ConstructOperator constructor;
      constructor.UpdateTensorShape(origin_shape);

      auto reshape = gen_g.PushBack({gen_g.NewOpInst(prim::kPrimReshape->name()), matmul, CreateTuple(origin_shape)});
      auto allgather = gen_g.PushBack({NewAllGatherNode(ALL_GATHER, HCCL_WORLD_GROUP), reshape});
      auto reshape2 = gen_g.PushBack({gen_g.NewOpInst(prim::kPrimReshape->name()), allgather, CreateTuple(des_shape)});
      std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(matmul, 1), std::make_pair(matmul, 2)};
      auto replace_graph = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
        std::make_pair(input_nodes, reshape2));
      MS_LOG(INFO) << "StepReplaceGraph " << expect_matmul->ToString();
      StepReplaceGraph(replace_graph, expect_matmul->cast<CNodePtr>());
      return;

    } else {
      auto allgather = gen_g.PushBack({NewAllGatherNode(ALL_GATHER, HCCL_WORLD_GROUP), matmul});
      // split
      int64_t split_count = devices;
      Attr split_axis_attr = std::make_pair(AXIS, MakeValue(0));
      Attr split_count_attr = std::make_pair(OUTPUT_NUM, MakeValue(split_count));
      OperatorAttrs split_attrs = {split_axis_attr, split_count_attr};
      auto split = gen_g.PushBack({gen_g.NewOpInst(SPLIT, split_attrs), allgather});

      // tuple get item and make tuple
      std::vector<AnfNodePtr> maketuple_inputs;
      maketuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
      for (int64_t i = 0; i < split_count; ++i) {
        auto tuple_get_item = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), split, CreatInt64Imm(i)});
        maketuple_inputs.push_back(tuple_get_item);
      }
      auto maketuple = gen_g.PushBack(maketuple_inputs);

      // concat
      Attr concat_axis_attr = std::make_pair(AXIS, MakeValue(1));
      OperatorAttrs concat_attrs = {concat_axis_attr};
      auto concat = gen_g.PushBack({gen_g.NewOpInst(CONCAT, concat_attrs), maketuple});

      std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(matmul, 1), std::make_pair(matmul, 2)};
      auto replace_graph = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
        std::make_pair(input_nodes, concat));
      MS_LOG(INFO) << "StepReplaceGraph " << expect_matmul->DebugString();
      StepReplaceGraph(replace_graph, expect_matmul->cast<CNodePtr>());
      return;
    }
  }
  return;
}

bool StepAssignedParallel(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager, size_t device_num,
                          size_t rank_id, bool sapp) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  // control whether use model_parallel mode
  if (device_num == 0 || device_num > 8) {
    MS_LOG(EXCEPTION) << "Error: device_num is <= 0 or > 8.";
    return false;
  }

  MSLogTime msTime;
  msTime.Start();
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpGraph(root, std::string("step_assigned_parallel_begin"));
  }
#endif
  MS_LOG(INFO) << "Now entering step assigned parallel";
  TOTAL_OPS = 0;
  AnfNodePtr ret = root->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

  if (ParallelInit(rank_id, device_num) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Parallel init failed";
  }

  MarkForwardCNode(root);

  if (sapp) {
    CostModelContext::GetInstance()->set_rp_matmul_mem_coef(1);
    if (ParallelStrategyRecSearch(all_nodes, root, rank_id, device_num) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Auto-parallel strategy search failed when using RP searching mode";
    }
    root->set_flag(AUTO_PARALLEL_RUN_ONCE_ONLY, true);
  }

  InitRefMap(root);
  // extract shape and strategy, set operator_info
  ExtractGraphInformation(all_nodes);

  MS_LOG(INFO) << "Now Assigned insert AllReduce opsl";

  if (!InsertAllReduceOps(all_nodes, root, device_num)) {
    MS_LOG(EXCEPTION) << "Assigned insert AllReduce ops failed.";
  }
  if (!InsertAllReduceOpsForFFN(all_nodes, root, device_num)) {
    MS_LOG(EXCEPTION) << "Assigned insert AllReduce ops failed.";
  }
  if (!ModifyReshapeOps(all_nodes, root, device_num)) {
    MS_LOG(EXCEPTION) << "Modify Reshape Ops failed.";
  }
  if (!ModifyMakeTupleOps(all_nodes, root, device_num)) {
    MS_LOG(EXCEPTION) << "Modify Reshape Ops failed.";
  }
  if (!ModifySoftmaxReshapeOps(all_nodes, root, device_num)) {
    MS_LOG(EXCEPTION) << "Modify Reshape Ops failed.";
  }

  ReplaceGatherOps(all_nodes, device_num);
  FixReturnRedistribution(root, device_num);
  DoParameterSliceShape(root, rank_id);
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    DumpGraph(root, std::string("step_assigned_parallel_end"));
  }
#endif

  msTime.End();
  uint64_t time = msTime.GetRunTimeUS();

  MS_LOG(INFO) << "Now leaving step assigned parallel, used time: " << time << " us";

  return true;
}

}  // namespace parallel
}  // namespace mindspore
