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

#include "frontend/parallel/pipeline_transformer/pipeline_interleave.h"
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include <unordered_set>
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/arithmetic_ops.h"
#include "frontend/optimizer/irpass/updatestate_eliminate.h"
#include "include/frontend/parallel/graph_costmodel.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/parameter_manager.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "ir/func_graph_cloner.h"
#include "include/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"
#include "frontend/parallel/parallel_node_check.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace parallel {

void PipelinePostProcess::MoveSharedParamMirrorOutCall(const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto crecv = node->cast<CNodePtr>();
    if (!crecv->HasPrimalAttr(PIPELINE_PARAM)) {
      continue;
    }
    auto param_node = crecv->user_data<AnfNode>(INPUT_PARAM);
    MS_EXCEPTION_IF_NULL(param_node);
    auto param = param_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);

    auto accu_parameter = FindGradAccuParameter(root_->parameters(), param->name());
    if (!accu_parameter) {
      continue;
    }
    auto accu_param_users = manager_->node_users()[accu_parameter];
    CNodePtr mirror_micro_node = nullptr;
    for (const auto &accu_param_user : accu_param_users) {
      if (!IsPrimitiveCNode(accu_param_user.first, prim::kPrimMirrorMicroStep)) {
        continue;
      }
      mirror_micro_node = accu_param_user.first->cast<CNodePtr>();
    }
    if (mirror_micro_node) {
      auto new_mirror_node = MoveSingeMirrorOutCallFunc(mirror_micro_node);
      MS_EXCEPTION_IF_NULL(new_mirror_node);
      MS_EXCEPTION_IF_NULL(new_mirror_node->cast<CNodePtr>());
      new_mirror_node->cast<CNodePtr>()->AddAttr(kPipelineSendSharedParam, MakeValue(true));
      param->set_user_data(kSharedParamMirrorNode, new_mirror_node);
    }
  }
}

void PipelinePostProcess::EliminateUpdateStateMakeTupleWithUselessLoadNode() {
  const auto &main_graph_nodes = TopoSort(main_graph_->get_return(), SuccDeeperSimple);
  for (const auto &node : main_graph_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      continue;
    }
    auto update_state = node->cast<CNodePtr>();
    auto new_update_state = opt::irpass::EliminateUpdateStateMakeTupleWithUselessLoadNode(update_state);
    if (new_update_state != nullptr) {
      MS_LOG(DEBUG) << "Replace UpdateState from update_state: " << update_state->DebugString()
                    << "to new_update_state: " << new_update_state->DebugString();
      (void)manager_->Replace(update_state, new_update_state);
    }
  }
}

void PipelinePostProcess::RemoveUselessOriginSharedCell() {
  EliminateUpdateStateMakeTupleWithUselessLoadNode();
  const auto &main_graph_nodes = TopoSort(main_graph_->get_return(), SuccDeeperSimple);
  for (const auto &node : main_graph_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<FuncGraph>(cnode->input(0))) {
      continue;
    }
    auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    if (fg != shared_cell_) {
      continue;
    }
    const auto &node_users_map = manager_->node_users();
    if (node_users_map.count(cnode) == 0) {
      continue;
    }
    auto origin_cell_users = node_users_map.at(cnode);
    for (const auto &origin_cell_user_pair : origin_cell_users) {
      if (IsPrimitiveCNode(origin_cell_user_pair.first, prim::kPrimUpdateState)) {
        auto abs = origin_cell_user_pair.first->abstract();
        auto monad_node = NewValueNode(kUMonad);
        if (abs->isa<abstract::AbstractIOMonad>()) {
          monad_node = NewValueNode(kIOMonad);
        }
        (void)manager_->Replace(origin_cell_user_pair.first, monad_node);
      } else if (IsPrimitiveCNode(origin_cell_user_pair.first, prim::kPrimDepend) &&
                 origin_cell_user_pair.second == SIZE_TWO) {
        (void)manager_->Replace(origin_cell_user_pair.first,
                                origin_cell_user_pair.first->cast<CNodePtr>()->input(SIZE_ONE));
      } else {
        MS_LOG(EXCEPTION) << "There are still contains origin lazy inline call after graph partition.";
      }
    }
  }
}

std::vector<AnfNodePtr> PipelinePostProcess::PartitionVShapeChunkGraph(const std::vector<AnfNodePtr> &sends) {
  auto all_nodes = DeepScopedGraphSearch(main_graph_->get_return());
  auto node_users_map = manager_->node_users();
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto recv_c = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(recv_c);
    if (recv_c->HasPrimalAttr(PIPELINE_PARAM)) {
      continue;
    }
    auto is_v_shape = GetValue<bool>(recv_c->GetPrimalAttr(V_SHAPE));
    if (!is_v_shape) {
      continue;
    }
    auto recv_tag = GetValue<int64_t>(recv_c->GetPrimalAttr(ORDER));
    for (const auto &send : sends) {
      auto send_c = send->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(send_c);
      if (send_c->HasPrimalAttr(PIPELINE_PARAM)) {
        continue;
      }
      auto send_tag = GetValue<int64_t>(send_c->GetPrimalAttr(ORDER));
      if (recv_tag != send_tag) {
        continue;
      }
      auto node_users = node_users_map.at(node);
      for (const auto &node_user : node_users) {
        manager_->SetEdge(node_user.first, node_user.second, send_c->input(1));
      }
      break;
    }
  }

  std::vector<AnfNodePtr> out_input = {NewValueNode(prim::kPrimMakeTuple)};
  for (const auto &send : sends) {
    auto send_c = send->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(send_c);
    if (send_c->HasPrimalAttr(PIPELINE_PARAM)) {
      out_input.emplace_back(send);
      continue;
    }
    auto is_v_shape = GetValue<bool>(send_c->GetPrimalAttr(V_SHAPE));
    if (is_v_shape) {
      continue;
    }
    out_input.emplace_back(send);
  }
  return out_input;
}

void PipelinePostProcess::GraphPartition(const std::vector<AnfNodePtr> &all_nodes) {
  LabelInterleaveIndex();
  if (NeededHandleShardParam()) {
    MoveSharedParamMirrorOutCall(all_nodes);
  }
  std::vector<AnfNodePtr> send_ops;
  auto no_need_clone_stage = is_v_shape_ ? 0 : stage_num_ - 1;
  for (size_t i = 0; i < LongToSize(chunk_num_); ++i) {
    auto chunk_fg = shared_cell_;
    if (stage_ != no_need_clone_stage || i != LongToSize(chunk_num_ - 1)) {
      chunk_fg = BasicClone(shared_cell_);
      chunk_fg->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);
      manager_->AddFuncGraph(chunk_fg);
    }
    chunk_fg->set_attr(CHUNK, MakeValue(i));
    auto sends = PartitionChunkGraph(chunk_fg, i);
    send_ops.insert(send_ops.begin(), sends.begin(), sends.end());
  }
  auto make_tuple = CreateMakeTupleNode(main_graph_, send_ops);
  auto outputs = GetZeroOutputs(main_graph_);
  if (stage_ == no_need_clone_stage) {
    outputs = main_graph_->output();
  }
  std::vector<AnfNodePtr> out = {NewValueNode(prim::kPrimDepend), outputs, make_tuple};
  auto out_node = main_graph_->NewCNode(out);
  (void)manager_->Replace(main_graph_->output(), out_node);
  if (is_v_shape_) {
    auto out_input = PartitionVShapeChunkGraph(send_ops);
    if (out_input.size() > 1) {
      make_tuple->set_inputs(out_input);
    }
    return;
  }
  if (stage_ == stage_num_ - 1) {
    return;
  }
  RemoveUselessOriginSharedCell();
}

void PipelinePostProcess::HandleSendParam() {
  auto parameters = root_->parameters();
  auto node_users_map = manager_->node_users();
  auto nodes = DeepScopedGraphSearch(root_->get_return());
  for (auto &node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      continue;
    }
    auto param = cnode->input(1);
    if (IsPrimitiveCNode(param, prim::kPrimVirtualAssignAdd)) {
      MS_EXCEPTION_IF_NULL(param->cast<CNodePtr>());
      param = param->cast<CNodePtr>()->input(1);
    }
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (NeededHandleShardParam()) {
      auto base_shape = param_ptr->Shape();
      auto shape_ptr = dyn_cast<abstract::Shape>(base_shape);
      auto slice_shape = shape_ptr->shape();
      auto prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(prim);
      auto value = MakeValue(slice_shape);
      prim->set_attr(SHAPE, value);
      param_ptr->set_user_data(kPipelineSendSharedParam, std::make_shared<bool>(true));
      continue;
    }

    auto accu_parameter = FindGradAccuParameter(parameters, param_ptr->name());
    if (!accu_parameter) {
      continue;
    }
    auto accu_users = node_users_map.at(accu_parameter);
    AnfNodePtr share_node = nullptr;
    for (auto &user : accu_users) {
      auto user_node = user.first;
      while (IsSomePrimitiveList(user_node->cast<CNodePtr>(),
                                 {prim::kPrimMirrorMicroStep->name(), prim::kPrimMicroStepAllGather->name()})) {
        share_node = user_node;
        user_node = node_users_map.at(user_node).front().first;
      }
      if (share_node == nullptr) {
        continue;
      }
      auto base_shape = accu_parameter->Shape();
      auto shape_ptr = dyn_cast<abstract::Shape>(base_shape);
      MS_EXCEPTION_IF_NULL(shape_ptr);
      auto slice_shape = shape_ptr->shape();
      auto prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(prim);
      std::vector<ValuePtr> element;
      (void)std::transform(slice_shape.begin(), slice_shape.end(), std::back_inserter(element),
                           [](int elem) { return MakeValue(int64_t(elem)); });
      auto value = std::make_shared<ValueList>(element);
      prim->set_attr(SHAPE, value);
      manager_->SetEdge(cnode, 1, share_node);
      break;
    }
  }
}

void PipelinePostProcess::ElimGraphStage() {
  for (auto &fg : manager_->func_graphs()) {
    fg->set_stage(-1);
  }
}

void PipelinePostProcess::RemoveMonadNode(const FuncGraphPtr &fg, int64_t chunk) {
  auto all_nodes = DeepScopedGraphSearch(fg->get_return());
  auto node_users_map = manager_->node_users();
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    RemoveMonadNodeBetweenStage(cnode);
    auto abs = cnode->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto stage_info = cnode->user_data<NodeStageInfo>();
    if (stage_info == nullptr) {
      continue;
    }
    auto stage = stage_info->stage();
    auto cur_chunk = stage_info->chunk();
    if ((stage != stage_ && stage != -1) || cur_chunk != chunk) {
      auto node_users = node_users_map[node];
      for (auto &user_node : node_users) {
        auto monad_node = NewValueNode(kUMonad);
        monad_node->set_abstract(kUMonad->ToAbstract());
        if (abs->isa<abstract::AbstractIOMonad>()) {
          monad_node = NewValueNode(kIOMonad);
          monad_node->set_abstract(kIOMonad->ToAbstract());
        }
        (void)manager_->SetEdge(user_node.first, user_node.second, monad_node);
      }
    }
  }
}

bool PipelineInterleave::IsNoUpdateParameterStage(const int64_t stage) {
  auto parameters = root_->parameters();
  for (auto &parameter : parameters) {
    if (ParameterIsCloned(parameter)) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(parameter->cast<ParameterPtr>());
    auto param_info = parameter->cast<ParameterPtr>()->param_info();
    if (!param_info) {
      continue;
    }
    auto stage_set = parameter_color_map_.at(parameter);
    auto requires_grad = param_info->requires_grad();
    if (requires_grad && stage_set.count(stage)) {
      return false;
    }
  }
  return true;
}

void PipelineInterleave::FreezeGradient() {
  auto node_users_map = manager_->node_users();
  if (IsNoUpdateParameterStage(stage_) && is_train_) {
    if (stage_ == 0) {
      MS_LOG(EXCEPTION) << "Stage 0 should has at least 1 trainable parameter. but got none. "
                        << "One possible cause is that the @lazy_inline decorator is misplaced.";
    }
    root_->set_flag(NO_UPDATE, true);
    auto nodes = root_->nodes();
    for (auto &node : nodes) {
      if (!IsPrimitiveCNode(node, prim::kPrimJ)) {
        continue;
      }
      auto node_users = node_users_map.at(node);
      auto grad_users = node_users_map.at(node_users.front().first);
      for (auto &grad_user : grad_users) {
        auto user_node = grad_user.first->cast<CNodePtr>();
        if (!IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
          continue;
        }
        auto index = GetTupleGetItemIndex(user_node);
        if (index != 1) {
          continue;
        }
        auto temp = node_users_map.at(user_node).front().first;
        auto out = root_->output();
        MS_EXCEPTION_IF_NULL(out);
        std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), out, temp};
        auto new_node = root_->NewCNode(depend_input);
        manager_->Replace(out, new_node);
        break;
      }
      break;
    }
    auto tftEnv = common::GetEnv("MS_ENABLE_TFT");
    constexpr std::string_view optUCE = "UCE:1";
    constexpr std::string_view optTTP = "TTP:1";
    constexpr std::string_view optARF = "ARF:1";
    bool enableTFT =
      !tftEnv.empty() && (tftEnv.find(optUCE) != std::string::npos || tftEnv.find(optTTP) != std::string::npos ||
                          tftEnv.find(optARF) != std::string::npos);
    for (auto &node : nodes) {
      if (!IsPrimitiveCNode(node, prim::kPrimNPUGetFloatStatusV2) && !IsTFTAllReduce(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      MS_EXCEPTION_IF_NULL(root_->output());
      auto out_cnode = root_->output()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_cnode);
      auto grads = out_cnode->input(INDEX_TWO);
      std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), cnode->input(1), grads};
      auto new_node = root_->NewCNode(depend_input);
      new_node->set_abstract(cnode->input(1)->abstract());
      manager_->Replace(cnode->input(1), new_node);
      if (!enableTFT) {
        break;
      }
    }
  }
}

static AnfNodePtr GetDout(const AnfNodePtr &node, const NodeUsersMap &node_users_map) {
  auto node_usrs = node_users_map.at(node);
  for (auto &node_user_pair : node_usrs) {
    auto node_usr = node_user_pair.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(node_usr, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto index = GetTupleGetItemIndex(node_usr);
    if (index != 1) {
      continue;
    }
    auto get_item_usrs = node_users_map.at(node_usr);
    if (get_item_usrs.size() != 1) {
      MS_LOG(WARNING) << "Get Multi grad usrs. Use first.";
    }
    return get_item_usrs.begin()->first;
  }
  return nullptr;
}

static bool NeedAttach(const FuncGraphPtr &root) {
  if (root->has_flag(HAS_ATTACHED)) {
    return false;
  }
  auto manager = root->manager();
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kAutoParallel && parallel_mode != kSemiAutoParallel) {
    return false;
  }
  bool cell_reuse = false;
  for (auto &fg : manager->func_graphs()) {
    if (fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      cell_reuse = true;
      break;
    }
  }
  auto stage_num = g_device_manager->stage_num();
  if (!cell_reuse || stage_num <= 1) {
    return false;
  }
  return true;
}

static AnfNodePtr GetUserNode(const AnfNodePtr &node, const NodeUsersMap &node_users_map) {
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<FuncGraph>(cnode->input(0))) {
    return nullptr;
  }
  auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(graph);
  auto sub_graph_output = graph->output();
  if (!IsPrimitiveCNode(sub_graph_output, prim::kPrimMakeTuple)) {
    return nullptr;
  }
  auto csub_graph_output = sub_graph_output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(csub_graph_output);
  MS_EXCEPTION_IF_NULL(csub_graph_output->input(1));
  if (!IsPrimitiveCNode(csub_graph_output->input(1), prim::kPrimReceive)) {
    return nullptr;
  }
  auto recv = csub_graph_output->input(1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(recv);
  if (recv->HasPrimalAttr(FREEZE)) {
    auto freeze_v = recv->GetPrimalAttr(FREEZE);
    if (GetValue<bool>(freeze_v)) {
      return nullptr;
    }
  }
  auto call_node_input = cnode->input(1);
  if (!IsValueNode<tensor::Tensor>(call_node_input)) {
    return nullptr;
  }
  auto call_node_users = node_users_map.at(node);
  if (call_node_users.size() != 1) {
    return nullptr;
  }
  auto user_node = call_node_users.begin()->first;
  if (!IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
    return nullptr;
  }
  return user_node;
}

bool IsolatedNodeAttach(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  if (!NeedAttach(root)) {
    return false;
  }
  root->set_flag(HAS_ATTACHED, true);
  auto manager = root->manager();
  auto ret_after = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = DeepScopedGraphSearch(ret_after);
  const auto &node_users_map = manager->node_users();
  std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple)};
  FuncGraphPtr main_graph;
  FuncGraphPtr grad_graph;
  for (auto &node : all_nodes) {
    const auto &user_node = GetUserNode(node, node_users_map);
    if (user_node == nullptr) {
      continue;
    }
    auto get_item_usrs = node_users_map.at(user_node);
    std::vector<AnfNodePtr> addn_input = {NewValueNode(prim::kPrimAddN)};
    main_graph = node->func_graph();
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (auto &get_item_usr_pair : get_item_usrs) {
      auto get_item_usr = get_item_usr_pair.first;
      auto grad_node = GetDout(get_item_usr, node_users_map);
      MS_EXCEPTION_IF_NULL(grad_node);
      if (grad_graph == nullptr) {
        grad_graph = grad_node->func_graph();
      } else {
        if (grad_graph != grad_node->func_graph()) {
          MS_LOG_WITH_NODE(EXCEPTION, cnode)
            << "Got Wrong Grad graph when attached Receive's grad, Maybe don't use lazy inline.";
        }
      }
      std::vector<AnfNodePtr> new_get_item_input = {NewValueNode(prim::kPrimTupleGetItem), grad_node,
                                                    NewValueNode(MakeValue(SizeToLong(get_item_usr_pair.second)))};
      auto new_get_item = grad_graph->NewCNode(new_get_item_input);
      addn_input.emplace_back(new_get_item);
    }
    AnfNodePtr temp = (addn_input.size() > SIZE_TWO) ? grad_graph->NewCNode(addn_input) : addn_input.at(1);
    std::vector<AnfNodePtr> send_grad_fn_input = {NewValueNode(prim::kPrimTupleGetItem), node,
                                                  NewValueNode(MakeValue(int64_t(1)))};
    auto send_grad_fn = main_graph->NewCNode(send_grad_fn_input);
    auto call_grad_node = grad_graph->NewCNode({send_grad_fn, temp});
    std::vector<AnfNodePtr> call_grad_get_item_input = {NewValueNode(prim::kPrimTupleGetItem), call_grad_node,
                                                        NewValueNode(MakeValue(int64_t(1)))};
    auto call_grad_get_item = grad_graph->NewCNode(call_grad_get_item_input);
    make_tuple_input.emplace_back(call_grad_get_item);
  }
  if (make_tuple_input.size() <= 1) {
    return false;
  }
  auto make_tuple = grad_graph->NewCNode(make_tuple_input);
  if (root->has_flag(NO_UPDATE)) {
    manager->Replace(grad_graph->output(), make_tuple);
    return true;
  }
  std::vector<AnfNodePtr> attach_node_input = {NewValueNode(prim::kPrimDepend), grad_graph->output(), make_tuple};
  auto attach_node = grad_graph->NewCNode(attach_node_input);
  manager->Replace(grad_graph->output(), attach_node);
  return true;
}
}  // namespace parallel
}  // namespace mindspore
