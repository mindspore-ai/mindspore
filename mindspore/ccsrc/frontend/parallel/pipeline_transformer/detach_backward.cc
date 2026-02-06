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

#include "frontend/parallel/pipeline_transformer/detach_backward.h"
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include "ir/func_graph.h"
#include "ir/core_ops_primitive.h"
#include "ir/func_graph_cloner.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/framework_ops.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace parallel {
constexpr int64_t kSizeTwo = 2;
std::vector<PPInfo> InferNeedDetachInfo(int64_t stage, int64_t micro_size) {
  auto stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  std::vector<PPInfo> need_detach_info;
  auto offset = common::GetEnv("ZBV_offset");
  int64_t offset_int = 0;
  if (offset.empty()) {
    offset = "0";
  }
  (void)StringToInt(&offset, &offset_int);
  // nB1W1
  for (int64_t i = 0; i < stage_num - stage - 1; ++i) {
    PPInfo info = {1, i};
    need_detach_info.emplace_back(info);
  }
  // phase 0
  for (int64_t i = 0; i < (stage + 1) / kSizeTwo - offset_int; ++i) {
    PPInfo info = {1, micro_size - 1 - i};
    need_detach_info.emplace_back(info);
  }
  // phase 1
  int64_t detach_num = stage_num - (stage + 1) / kSizeTwo;
  for (int64_t i = 0; i < detach_num - offset_int; ++i) {
    PPInfo info = {0, micro_size - 1 - i};
    need_detach_info.emplace_back(info);
  }
  return need_detach_info;
}

AnfNodePtr GetAssignAddRealInput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto assign_add_input = node;
  auto is_redistribe_node = [](const AnfNodePtr &n) {
    auto prim = GetCNodePrimitive(n);
    if (prim == nullptr) {
      return false;
    }
    const auto &prim_name = prim->instance_name();
    if (prim_name.find(parallel::REDISTRIBUTION_OP) != std::string::npos) {
      return true;
    }
    return false;
  };
  auto bypass_op = [is_redistribe_node](const AnfNodePtr &n) {
    return IsOneOfPrimitiveCNode(n, {prim::kPrimCast, prim::kPrimTranspose, prim::kPrimReshape, prim::kPrimDepend,
                                     prim::kPrimUpdateState}) ||
           is_redistribe_node(n);
  };
  while (bypass_op(assign_add_input)) {
    auto cnode = assign_add_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    assign_add_input = cnode->input(1);
  }
  return assign_add_input;
}

void DetachBackward::Init() {
  MS_EXCEPTION_IF_NULL(manager_);
  for (const auto &fg : manager_->func_graphs()) {
    // get closure graph
    if (fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      closure_graphs_.insert(fg);
    }
  }
  if (closure_graphs_.empty()) {
    MS_LOG(WARNING) << "Can't find cell reuse graph, detach backward pass is invalid.";
  }
  GetChunkNumMicroSize();
  need_detach_info_ = InferNeedDetachInfo(stage_, micro_size_);
}

bool DetachBackward::IsNeedDetach(int64_t chunk, int64_t micro) {
  return std::any_of(need_detach_info_.begin(), need_detach_info_.end(),
                     [chunk, micro](const auto &info) { return info.chunk == chunk && info.micro == micro; });
}

void DetachBackward::GetChunkNumMicroSize() {
  auto all_nodes = TopoSort(root_->get_return(), SuccDeeperSimple);
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->HasPrimalAttr(CHUNK) || !cnode->HasPrimalAttr(MICRO)) {
      continue;
    }
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    chunk_num_ = (chunk + 1) > chunk_num_ ? (chunk + 1) : chunk_num_;
    auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
    micro_size_ = (micro + 1) > micro_size_ ? (micro + 1) : micro_size_;
  }
}

std::vector<size_t> DetachBackward::DetachDxAndDwGraph(const FuncGraphPtr &fg, bool is_dw_fg,
                                                       const CNodePtr &partial_cnode,
                                                       std::vector<AnfNodePtr> *new_partial_inputs) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(fg->output());
  auto fg_output = fg->output()->cast<CNodePtr>();
  if (!IsPrimitiveCNode(fg_output, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION)
      << "Currently, it's not supported to detach the output of backward func_graphs that are not tuples. func_graph:"
      << fg->ToString();
  }
  std::vector<AnfNodePtr> dx_out_inputs;
  std::vector<AnfNodePtr> dw_out_inputs;
  std::vector<size_t> dw_index;
  auto fg_params = fg->parameters();
  auto num_diff = fg_params.size() + kSizeTwo - partial_cnode->inputs().size();
  for (size_t i = 1; i < fg_output->inputs().size(); ++i) {
    auto cur_input = fg_output->input(i);
    if (!IsPrimitiveCNode(cur_input, prim::kPrimDepend)) {
      dx_out_inputs.emplace_back(cur_input);
      continue;
    }
    auto depend_c = cur_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_c);
    if (!IsPrimitiveCNode(depend_c->input(kIndex2), prim::kPrimAssignAdd)) {
      dx_out_inputs.emplace_back(cur_input);
      continue;
    }
    auto assign_add_c = depend_c->input(kIndex2)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(assign_add_c);
    // case1: MatMul(Dw)->assign_add->depend
    auto assign_add_input = assign_add_c->input(kIndex2);
    assign_add_input = GetAssignAddRealInput(assign_add_input);
    MS_EXCEPTION_IF_NULL(assign_add_input);
    if (!IsPrimitiveCNode(assign_add_input, prim::kPrimMatMul) &&
        !IsPrimitiveCNode(assign_add_input, prim::kPrimMatMulExt) &&
        !IsPrimitiveCNode(assign_add_input, prim::kPrimTupleGetItem)) {
      dx_out_inputs.emplace_back(cur_input);
      continue;
    }
    auto dw_c = assign_add_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dw_c);
    // case2: GMM(Dw)->TupleGetItem->assign_add->depend
    if (IsPrimitiveCNode(assign_add_input, prim::kPrimTupleGetItem)) {
      auto get_item_c_input = dw_c->input(1);
      MS_EXCEPTION_IF_NULL(get_item_c_input);
      if (!IsPrimitiveCNode(get_item_c_input, prim::kPrimGroupedMatmul)) {
        dx_out_inputs.emplace_back(cur_input);
        continue;
      }
      MS_EXCEPTION_IF_NULL(get_item_c_input->cast<CNodePtr>());
      MS_EXCEPTION_IF_NULL(get_item_c_input->cast<CNodePtr>()->input(kIndex2));
      dw_c = get_item_c_input->cast<CNodePtr>()->input(kIndex2)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(dw_c);
      // Dw overlap depend
      while (IsPrimitiveCNode(dw_c, prim::kPrimDepend)) {
        dw_c = dw_c->input(kIndex1)->cast<CNodePtr>();
      }
    }
    dw_index.emplace_back(i - 1);
    dw_out_inputs.emplace_back(cur_input);
    size_t input_index = kIndex1;
    MS_EXCEPTION_IF_NULL(dw_c);
    if (dw_c->HasPrimalAttr(FORWARD_TRANSPOSE_B) && !GetValue<bool>(dw_c->GetPrimalAttr(FORWARD_TRANSPOSE_B))) {
      input_index = kIndex2;
    }
    dx_out_inputs.emplace_back(dw_c->input(input_index));
    if (is_dw_fg) {
      auto fg_new_param = std::make_shared<Parameter>(fg);
      fg_params.emplace_back(fg_new_param);
      manager_->SetEdge(dw_c, input_index, fg_new_param);
    }
  }
  auto no_used_index =
    HandleBwdGraphOutputs(std::make_pair(dx_out_inputs, dw_out_inputs), is_dw_fg, fg, fg_params, num_diff);
  for (size_t i = 2; i < partial_cnode->inputs().size(); ++i) {
    if (std::find(no_used_index.begin(), no_used_index.end(), i) == no_used_index.end()) {
      new_partial_inputs->emplace_back(partial_cnode->input(i));
    }
  }
  return dw_index;
}

AnfNodePtr DetachBackward::CreateMakeTuple(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &inputs) {
  MS_EXCEPTION_IF_NULL(fg);
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  make_tuple_inputs.insert(make_tuple_inputs.end(), inputs.begin(), inputs.end());
  auto make_tuple = fg->NewCNode(make_tuple_inputs);
  return make_tuple;
}

AnfNodePtr DetachBackward::CreateTupleGetItem(const FuncGraphPtr &fg, const AnfNodePtr &node, int64_t index) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> tuple_getitem_inputs = {NewValueNode(prim::kPrimTupleGetItem), node,
                                                  NewValueNode(MakeValue<int64_t>(index))};
  auto tuple_getitem = fg->NewCNode(tuple_getitem_inputs);
  return tuple_getitem;
}

std::vector<size_t> DetachBackward::HandleBwdGraphOutputs(
  const std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr> > &out_inputs, bool is_dw_fg, const FuncGraphPtr &fg,
  const std::vector<AnfNodePtr> &parameters, size_t num_diff) {
  auto output = fg->output();
  const auto &dx_out_inputs = out_inputs.first;
  if (!is_dw_fg && !dx_out_inputs.empty()) {
    auto make_tuple = CreateMakeTuple(fg, dx_out_inputs);
    manager_->Replace(output, make_tuple);
  }
  const auto &dw_out_inputs = out_inputs.second;
  if (is_dw_fg && !dw_out_inputs.empty()) {
    auto make_tuple = CreateMakeTuple(fg, dw_out_inputs);
    manager_->Replace(output, make_tuple);
    manager_->SetParameters(fg, parameters);
  }
  // Remove parameters not used after detach
  auto params = fg->parameters();
  std::vector<AnfNodePtr> parameter_used;
  std::vector<size_t> no_used_index;
  auto node_users_map = manager_->node_users();
  for (size_t i = 0; i < params.size(); ++i) {
    auto cur_param = params.at(i);
    if (i >= (params.size() - num_diff) && !is_dw_fg) {
      parameter_used.emplace_back(cur_param);
      continue;
    }
    const auto &iter = node_users_map.find(cur_param);
    if (iter == node_users_map.end()) {
      no_used_index.emplace_back(i + kIndex2);
      continue;
    }
    if (node_users_map.at(cur_param).size() == 0) {
      no_used_index.emplace_back(i + kIndex2);
      continue;
    }
    parameter_used.emplace_back(cur_param);
  }
  manager_->SetParameters(fg, parameter_used);
  return no_used_index;
}

size_t DetachBackward::HandleMonadNode(const FuncGraphPtr &dx_fg, const FuncGraphPtr &dw_fg, size_t partial_dw_size,
                                       std::vector<size_t> *dw_index) {
  auto params_num = dw_fg->parameters().size();
  auto input_size = partial_dw_size + dw_index->size();
  if (params_num < input_size) {
    MS_LOG(EXCEPTION) << "Dw fg's inputs num is wrong. Dw fg:" << dw_fg->ToString();
  }
  auto num_diff = params_num - input_size;
  if (num_diff != 0) {
    MS_EXCEPTION_IF_NULL(dx_fg);
    MS_EXCEPTION_IF_NULL(dx_fg->output());
    auto dx_fg_output = dx_fg->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dx_fg_output);
    auto dx_new_out_inputs = dx_fg_output->inputs();
    auto dx_fg_params = dx_fg->parameters();
    for (size_t i = 0; i < num_diff; ++i) {
      auto cur_param = dx_fg_params.at(dx_fg_params.size() - num_diff + i);
      dx_new_out_inputs.emplace_back(cur_param);
      dw_index->insert(dw_index->begin() + i, dx_new_out_inputs.size() - kIndex2);
    }
    auto dx_new_out = dx_fg->NewCNode(dx_new_out_inputs);
    manager_->Replace(dx_fg_output, dx_new_out);
  }
  return num_diff;
}

CNodePtr DetachBackward::CreateDwCallNode(const NodeUsersMap &node_users_map, const CNodePtr &dx_call_node,
                                          const CNodePtr &closure_call_node, std::vector<size_t> dw_index) {
  auto cur_fg = closure_call_node->func_graph();
  auto dw_func = CreateTupleGetItem(cur_fg, closure_call_node, INT64_THREE);
  std::vector<AnfNodePtr> dw_call_inputs = {dw_func};
  auto dx_call_users = node_users_map.at(dx_call_node);
  for (size_t i = 0; i < dw_index.size(); ++i) {
    bool has_item = false;
    for (const auto &dx_call_user : dx_call_users) {
      auto dx_user_node = dx_call_user.first->cast<CNodePtr>();
      if (!IsPrimitiveCNode(dx_user_node, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto index = GetTupleGetItemIndex(dx_user_node);
      if (LongToSize(index) != dw_index[i]) {
        continue;
      }
      has_item = true;
      dw_call_inputs.emplace_back(dx_user_node);
    }
    if (!has_item) {
      auto manual_get_item = CreateTupleGetItem(cur_fg, dx_call_node, dw_index[i]);
      dw_call_inputs.emplace_back(manual_get_item);
    }
  }
  auto dw_call_node = cur_fg->NewCNode(dw_call_inputs);
  return dw_call_node;
}

void DetachBackward::HandleDataDependency(const std::vector<size_t> &dw_index, const FuncGraphPtr &fg,
                                          size_t num_diff) {
  auto node_users_map = manager_->node_users();
  auto fg_users = fg->func_graph_cnodes_index();
  for (const auto &fg_user : fg_users) {
    CNodePtr dx_call_node;
    auto closure_call_node = fg_user.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(closure_call_node);
    auto closure_call_attrs = closure_call_node->attrs();
    if (!closure_call_node->HasAttr(CHUNK) || !closure_call_node->HasAttr(MICRO)) {
      MS_LOG(EXCEPTION) << "closure call node doesn't have chunk or micro information";
    }
    auto cur_chunk = GetValue<int64_t>(closure_call_attrs[CHUNK]);
    auto cur_micro = GetValue<int64_t>(closure_call_attrs[MICRO]);
    auto is_need_detach = IsNeedDetach(cur_chunk, cur_micro);
    if (!is_need_detach) {
      continue;
    }
    auto closure_call_users = node_users_map.at(closure_call_node);
    for (const auto &closure_call_user : closure_call_users) {
      if (!IsPrimitiveCNode(closure_call_user.first, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto index = GetTupleGetItemIndex(closure_call_user.first->cast<CNodePtr>());
      if (index != 1) {
        continue;
      }

      // dx_func
      auto dx_func_users = node_users_map.at(closure_call_user.first);
      for (const auto &dx_func_user : dx_func_users) {
        if (dx_func_user.second != 0) {
          continue;
        }
        dx_call_node = dx_func_user.first->cast<CNodePtr>();
        (void)manager_->SetEdge(closure_call_user.first, kIndexTwo, NewValueNode(MakeValue<int64_t>(kIndexTwo)));
        break;
      }
    }
    MS_EXCEPTION_IF_NULL(dx_call_node);
    auto dw_call_node = CreateDwCallNode(node_users_map, dx_call_node, closure_call_node, dw_index);
    auto dw_call_fg = dw_call_node->func_graph();
    for (size_t i = 1 + num_diff; i < dw_call_node->inputs().size(); ++i) {
      auto cur_input = dw_call_node->input(i);
      auto get_item = CreateTupleGetItem(dw_call_fg, dw_call_node, i - 1 - num_diff);
      auto cur_input_users = node_users_map.at(cur_input);
      for (const auto &cur_input_user : cur_input_users) {
        manager_->SetEdge(cur_input_user.first, cur_input_user.second, get_item);
      }
    }
  }
}

void DetachBackward::HandleClosureGraph(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(fg->output());
  auto closure_output = fg->output()->cast<CNodePtr>();
  if (!IsPrimitiveCNode(closure_output, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION)
      << "Currently, it's not supported to detach the output of backward func_graphs that are not tuples. func_graph:"
      << fg->ToString();
  }
  if (closure_output->inputs().size() < kIndex3) {
    MS_LOG(EXCEPTION) << "Currently, closure graph's outputs size less than 3 is not supported. closure_graph:"
                      << fg->ToString();
  }
  if (!IsPrimitiveCNode(closure_output->input(kIndex2), prim::kPrimPartial)) {
    MS_LOG(EXCEPTION) << "The Partial node was not found in the output of the closure graph. closure_graph:"
                      << fg->ToString();
  }
  auto partial_node = closure_output->input(kIndex2)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(partial_node);
  auto bwd_fg = GetValueNode<FuncGraphPtr>(partial_node->input(kIndex1));
  MS_EXCEPTION_IF_NULL(bwd_fg);
  AdapteDwOverlap(bwd_fg);
  // detach dw fg
  auto dw_fg = BasicClone(bwd_fg);
  manager_->AddFuncGraph(dw_fg);
  std::vector<AnfNodePtr> partial_dw_inputs = {NewValueNode(prim::kPrimPartial), NewValueNode(dw_fg)};
  auto dw_index = DetachDxAndDwGraph(dw_fg, true, partial_node, &partial_dw_inputs);
  auto partial_dw_node = fg->NewCNode(partial_dw_inputs);

  // detach dx fg
  auto dx_fg = BasicClone(bwd_fg);
  manager_->AddFuncGraph(dx_fg);
  std::vector<AnfNodePtr> partial_dx_inputs = {NewValueNode(prim::kPrimPartial), NewValueNode(dx_fg)};
  (void)DetachDxAndDwGraph(dx_fg, false, partial_node, &partial_dx_inputs);
  auto partial_dx_node = fg->NewCNode(partial_dx_inputs);

  // Replace closure fg output
  std::vector<AnfNodePtr> closure_out_inputs = {closure_output->input(kIndex1), closure_output->input(kIndex2),
                                                partial_dx_node, partial_dw_node};
  auto new_output = CreateMakeTuple(fg, closure_out_inputs);
  MS_EXCEPTION_IF_NULL(new_output);
  manager_->Replace(closure_output, new_output);

  auto num_diff = HandleMonadNode(dx_fg, dw_fg, partial_dw_inputs.size() - kIndex2, &dw_index);

  // Handle dx dw data dependency
  HandleDataDependency(dw_index, fg, num_diff);
}

void DetachBackward::AdapteDwOverlap(const FuncGraphPtr &fg) {
  static const std::array<const char *, 5> kAttrs = {"matmul_grad_depend1", "matmul_grad_depend2",
                                                     "matmul_grad_depend3", "recompute_depend1", "recompute_depend2"};
  auto nodes = fg->nodes();
  for (const auto &node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (const auto &attr : kAttrs) {
      if (cnode->HasAttr(attr)) {
        manager_->Replace(cnode, cnode->input(1));
        break;
      }
    }
  }
}

void DetachBackward::Run() {
  for (const auto &fg : closure_graphs_) {
    HandleClosureGraph(fg);
  }
}
}  // namespace parallel
}  // namespace mindspore
