/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "base/base.h"
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/arithmetic_ops.h"
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
#include "frontend/parallel/tensor_layout/shared_parameter.h"
#include "frontend/parallel/strategy_utils.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "include/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"
#include "frontend/parallel/parallel_node_check.h"
#include "include/utils/ir_dump/anf_ir_dump_interface.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
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
#include "frontend/parallel/pipeline_transformer/pipeline_transformer_utils.h"

namespace mindspore {
namespace parallel {

extern mindspore::HashMap<int64_t, int64_t> send_tag_map;
extern mindspore::HashMap<int64_t, int64_t> recv_tag_map;

std::vector<AnfNodePtr> PipelineTransformer::FetchSend(const AnfNodePtr &node, bool pipeline_param,
                                                       bool single_pipeline_end, size_t end_index) {
  std::vector<AnfNodePtr> depends;
  AnfNodePtr send_input;
  if (pipeline_param) {
    auto param = node->user_data<AnfNode>(INPUT_PARAM);
    MS_EXCEPTION_IF_NULL(param);
    auto params = shared_cell_->parameters();
    auto iter = std::find(params.begin(), params.end(), param);
    if (iter != params.end()) {
      auto input_pos = std::distance(params.begin(), iter) + 1;
      auto &front = shared_cell_users_.front();
      MS_EXCEPTION_IF_NULL(front);
      const auto &user = front->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user);
      send_input = user->input(input_pos);
    } else {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      send_input = cnode->input(INDEX_ONE);
    }
    MS_EXCEPTION_IF_NULL(send_input);
    auto value = MakeValue(int64_t(0));
    (void)(depends.emplace_back(GenNewSendFromOld(node, send_input, value)));
    return depends;
  }
  for (auto &user : shared_cell_users_) {
    auto cuser = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cuser);
    auto value = shared_cell_users_.size() > 1 ? cuser->GetPrimalAttr(MICRO) : MakeValue(int64_t(0));
    MS_EXCEPTION_IF_NULL(value);
    send_input = single_pipeline_end ? user : CreateTupleGetItemNode(main_graph_, user, end_index);
    (void)(depends.emplace_back(GenNewSendFromOld(node, send_input, value)));
  }
  return depends;
}

void PipelineTransformer::HandleGraphOutputs(const std::vector<AnfNodePtr> &nodes) {
  std::vector<AnfNodePtr> pipeline_params;
  std::vector<AnfNodePtr> pipeline_ends;
  SeparatePipelineParamBorder(nodes, true, &pipeline_params, &pipeline_ends);
  std::vector<AnfNodePtr> sends;
  SetNodeAbstract(pipeline_ends);
  // Create root graph output before modify subgraph(shared cell).
  // This process order is crucial when the output of subgraph is directly used as root graph.
  auto zero_outputs = GetZeroOutputs(main_graph_);
  MS_EXCEPTION_IF_NULL(zero_outputs);
  size_t ends_size = pipeline_ends.size();
  bool single_pipeline_end = ends_size == 1;
  if (single_pipeline_end) {
    auto &depend = pipeline_ends.front();
    const auto &cdepend = depend->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cdepend);
    (void)manager_->Replace(shared_cell_->output(), cdepend->input(INDEX_ONE));
  } else {
    std::vector<AnfNodePtr> rets;
    (void)std::transform(pipeline_ends.begin(), pipeline_ends.end(), std::back_inserter(rets),
                         [](const AnfNodePtr &depend) {
                           const auto &cdepend = depend->cast<CNodePtr>();
                           MS_EXCEPTION_IF_NULL(cdepend);
                           return cdepend->input(INDEX_ONE);
                         });
    auto out = CreateMakeTupleNode(shared_cell_, rets);
    (void)manager_->Replace(shared_cell_->output(), out);
  }
  for (auto &node : pipeline_params) {
    auto params = FetchSend(node, true, false, 0);
    if (is_train_) {
      (void)std::copy(params.begin(), params.end(), std::back_inserter(sends));
    }
  }

  for (size_t i = 0; i < ends_size; i++) {
    auto node = pipeline_ends[i];
    auto ends = FetchSend(node, false, single_pipeline_end, i);
    (void)std::copy(ends.begin(), ends.end(), std::back_inserter(sends));
  }

  auto make_tuple = CreateMakeTupleNode(main_graph_, sends);
  std::vector<AnfNodePtr> out = {NewValueNode(prim::kPrimDepend), zero_outputs, make_tuple};
  auto out_node = main_graph_->NewCNode(out);
  out_node->set_abstract(zero_outputs->abstract());
  (void)manager_->Replace(main_graph_->output(), out_node);
}

AnfNodePtr PipelineTransformer::GenNewRecvFromOld(const AnfNodePtr &node, const AnfNodePtr &input,
                                                  const ValuePtr &value) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto src_rank_ptr = cnode->user_data<int64_t>(SRC_RANK);
  MS_EXCEPTION_IF_NULL(src_rank_ptr);
  auto src_rank = *src_rank_ptr;
  auto recv_tag = recv_tag_map[src_rank];
  recv_tag_map[src_rank]++;
  auto dtype = node->user_data<Type>(SLICE_DTYPE);
  auto slice_shape_ptr = cnode->user_data<Shape>(SLICE_SHAPE);
  MS_EXCEPTION_IF_NULL(slice_shape_ptr);
  auto slice_shape = *slice_shape_ptr;
  auto shape = ShapeDimensionsToValueList(slice_shape);
  Attr attr_tag = std::make_pair(SR_TAG, MakeValue(recv_tag));
  Attr attr_rank = std::make_pair(SRC_RANK, MakeValue(src_rank));
  Attr attr_shape = std::make_pair(SHAPE, shape);
  Attr attr_dtype = std::make_pair(DTYPE, dtype);
  Attr attr_group = std::make_pair(GROUP, MakeValue(world_group_));
  Attr attr_group_back = std::make_pair(GROUP_BACK, MakeValue(world_group_));
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_shape, attr_dtype, attr_group, attr_group_back};

  std::vector<AnfNodePtr> recv_input = {input};
  auto recv = CreateCNodeByInputsAndAttr(main_graph_, RECEIVE, RECEIVE, recv_input, attrs);
  auto tensor_layout = node->user_data<TensorLayout>();
  if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
    auto abstract_clone = node->abstract()->Clone();
    MS_EXCEPTION_IF_NULL(abstract_clone);
    recv->set_user_data<AnfNode>(PIPELINE_PARAM, recv_input[INDEX_ZERO]);
    recv->AddPrimalAttr(PIPELINE_PARAM, value);
    recv_input[INDEX_ZERO]->set_abstract(abstract_clone);
    recv_input[INDEX_ZERO]->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(*tensor_layout));
  } else {
    recv->AddPrimalAttr(PIPELINE_BEGIN, value);
  }
  auto abstract_clone = node->abstract()->Clone();
  MS_EXCEPTION_IF_NULL(abstract_clone);
  recv->set_abstract(abstract_clone);

  recv->AddPrimalAttr(MICRO, value);
  if (cnode->HasPrimalAttr(FREEZE)) {
    recv->AddPrimalAttr(FREEZE, cnode->GetPrimalAttr(FREEZE));
  }
  recv->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(*tensor_layout));
  recv->set_user_data<OperatorInfo>(node->user_data<OperatorInfo>());
  return recv;
}

std::vector<AnfNodePtr> PipelineTransformer::FetchRecv(const AnfNodePtr &node, bool pipeline_param) {
  std::vector<AnfNodePtr> recvs;
  AnfNodePtr recv_input;
  AnfNodePtr recv;
  if (pipeline_param) {
    auto value = MakeValue(int64_t(0));
    auto param = node->user_data<AnfNode>(INPUT_PARAM);
    MS_EXCEPTION_IF_NULL(param);
    auto &front = shared_cell_users_.front();
    MS_EXCEPTION_IF_NULL(front);
    const auto &user = front->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user);
    auto params = shared_cell_->parameters();
    auto user_inputs = user->inputs();
    auto iter = std::find(user_inputs.begin(), user_inputs.end(), param);
    if (iter != user_inputs.end()) {
      auto input_pos = std::distance(user_inputs.begin(), iter);
      auto argu = params.at(input_pos - 1);
      manager_->SetEdge(node, 1, argu);
      node->set_user_data<AnfNode>(INPUT_PARAM, argu);
      recv_input = user->input(input_pos);
      recv = GenNewRecvFromOld(node, recv_input, value);
      for (auto &share_user : shared_cell_users_) {
        if (is_train_) {
          manager_->SetEdge(share_user, input_pos, recv);
        } else {
          manager_->SetEdge(share_user, input_pos, recv_input);
        }
      }
      node->set_user_data<bool>(ORIGIN_INPUT_IS_PARAM, std::make_shared<bool>(true));
    } else {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      recv_input = cnode->input(INDEX_ONE);
      recv = GenNewRecvFromOld(node, recv_input, value);
    }
    (void)(recvs.emplace_back(recv));
    return recvs;
  }
  for (auto &user : shared_cell_users_) {
    auto cuser = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cuser);
    auto value = shared_cell_users_.size() > 1 ? cuser->GetPrimalAttr(MICRO) : MakeValue(int64_t(0));
    MS_EXCEPTION_IF_NULL(value);
    if (enable_share_cell_ || !is_train_) {
      auto recv_tensor = TensorConstructUtils::CreateZerosTensor(kFloat16, {1});
      recv = GenNewRecvFromOld(node, NewValueNode(recv_tensor), value);
    } else {
      recv = GenNewRecvFromOld(node, virtual_param_, value);
    }
    (void)(recvs.emplace_back(recv));
  }
  return recvs;
}

void PipelineTransformer::ResetSharedCellParamAndArgu(
  const std::vector<std::vector<AnfNodePtr>> &pipeline_begins_fetched,
  const std::vector<AnfNodePtr> &newly_added_params, const std::vector<AnfNodePtr> &reserved_inputs) {
  // set shared_cell_ parameters, and call_input
  auto params = shared_cell_->parameters();
  auto ret = shared_cell_->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> searched_params;
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (auto &node : all_nodes) {
    if (node->isa<Parameter>()) {
      searched_params.push_back(node);
    }
  }
  std::set<size_t> reserved_param_index;
  std::vector<AnfNodePtr> new_params;
  std::vector<AnfNodePtr> monad_params;
  // set shared_cell_ parameters
  for (size_t i = 0; i < params.size(); i++) {
    auto param = params[i];
    if (std::find(searched_params.begin(), searched_params.end(), param) == searched_params.end()) {
      continue;
    }
    if (HasAbstractMonad(param)) {
      monad_params.push_back(param);
    } else {
      new_params.push_back(param);
    }
    (void)(reserved_param_index.insert(i));
  }
  (void)(new_params.insert(new_params.end(), newly_added_params.begin(), newly_added_params.end()));
  (void)(new_params.insert(new_params.end(), monad_params.begin(), monad_params.end()));
  MS_LOG(DEBUG) << "The shared cell origin params size is " << params.size() << ", new params size is "
                << new_params.size();
  manager_->SetParameters(shared_cell_, new_params);
  shared_cell_->set_fv_param_count(new_params.size());
  // set call inputs
  size_t user_index = 0;
  for (auto &user : shared_cell_users_) {
    auto cuser = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cuser);
    const auto &old_inputs = cuser->inputs();
    std::vector<AnfNodePtr> new_inputs{old_inputs.front()};
    std::vector<AnfNodePtr> monad_inputs;
    for (size_t i = 1; i < old_inputs.size(); i++) {
      if (reserved_param_index.find(i - 1) == reserved_param_index.end()) {
        continue;
      }
      auto old_input = old_inputs[i];
      if (HasAbstractMonad(old_input)) {
        monad_inputs.push_back(old_input);
      } else {
        new_inputs.push_back(old_input);
      }
    }
    auto newly_added_inputs = reserved_inputs;
    auto begins = pipeline_begins_fetched.at(user_index);
    (void)(newly_added_inputs.insert(newly_added_inputs.end(), begins.begin(), begins.end()));
    (void)(newly_added_inputs.insert(newly_added_inputs.end(), monad_inputs.begin(), monad_inputs.end()));
    (void)(new_inputs.insert(new_inputs.end(), newly_added_inputs.begin(), newly_added_inputs.end()));
    auto new_call = main_graph_->NewCNode(new_inputs);
    new_call->set_attrs(cuser->attrs());
    new_call->set_primal_attrs(cuser->primal_attrs());
    new_call->set_abstract(cuser->abstract());
    (void)manager_->Replace(user, new_call);
    user_index++;
  }
}

void PipelineTransformer::HandleGraphInputs(const std::vector<AnfNodePtr> &recv_ops) {
  std::vector<AnfNodePtr> pipeline_params;
  std::vector<AnfNodePtr> pipeline_begins;
  SeparatePipelineParamBorder(recv_ops, false, &pipeline_params, &pipeline_begins);

  // reserved inputs
  std::vector<AnfNodePtr> reserved_inputs;
  // pipeline_param whose input is a parameter
  std::vector<AnfNodePtr> pipeline_params_with_param_input;
  std::vector<AnfNodePtr> need_link_to_new_param;

  for (auto &node : pipeline_params) {
    auto recvs = FetchRecv(node, true);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->has_user_data(ORIGIN_INPUT_IS_PARAM)) {
      pipeline_params_with_param_input.push_back(node);
    } else {
      (void)(reserved_inputs.insert(reserved_inputs.end(), recvs.begin(), recvs.end()));
      need_link_to_new_param.push_back(node);
    }
  }
  (void)(need_link_to_new_param.insert(need_link_to_new_param.end(), pipeline_begins.begin(), pipeline_begins.end()));

  size_t begin_size = pipeline_begins.size();
  // The 0th dimension corresponds to shared_cell users
  // The first dimension corresponds to recvs
  // user0: recv0_0, recv0_1
  // user1: recv1_0, recv1_1
  size_t shared_cell_users_size = shared_cell_users_.size();
  std::vector<std::vector<AnfNodePtr>> pipeline_begins_fetched(shared_cell_users_size, std::vector<AnfNodePtr>());
  for (size_t i = 0; i < begin_size; i++) {
    auto node = pipeline_begins[i];
    auto begins = FetchRecv(node, false);
    for (size_t j = 0; j < shared_cell_users_size; j++) {
      pipeline_begins_fetched[j].push_back(begins.at(j));
    }
  }
  // relink pipeline_param_with_param_input's users to its input
  for (const auto &param : pipeline_params_with_param_input) {
    auto node_users_map = manager_->node_users();
    const auto &users = node_users_map[param];
    auto input = param->user_data<AnfNode>(INPUT_PARAM);
    MS_EXCEPTION_IF_NULL(input);
    for (const auto &user : users) {
      manager_->SetEdge(user.first, user.second, input);
    }
  }

  std::vector<AnfNodePtr> newly_added_params;
  // relink pipeline_param_without_param_input and pipeline_begins's users to new parameter
  for (const auto &node : need_link_to_new_param) {
    auto param = std::make_shared<Parameter>(shared_cell_);
    param->set_abstract(node->abstract()->Clone());
    newly_added_params.push_back(param);
    auto node_users_map = manager_->node_users();
    const auto &users = node_users_map[node];
    for (const auto &user : users) {
      manager_->SetEdge(user.first, user.second, param);
    }
  }
  ResetSharedCellParamAndArgu(pipeline_begins_fetched, newly_added_params, reserved_inputs);
}

AnfNodePtr PipelineTransformer::CreateTupleZeroTensor(const AnfNodePtr &node, size_t index) {
  std::vector<AnfNodePtr> temp_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  auto out_shapes = GetNodeShape(node);
  for (size_t ele = 0; ele < out_shapes.size(); ++ele) {
    temp_tuple_inputs.emplace_back(CreateZeroseOutput(node, ele));
  }
  auto temp_tuple = main_graph_->NewCNode(temp_tuple_inputs);
  SetMakeTupleAbstract(temp_tuple);
  return temp_tuple;
}

void PipelineTransformer::CutGraph() {
  world_group_ = GetWorldGroup();

  auto send_recv_shared_param = HandleSharedParameter();

  auto graph = enable_share_cell_ ? shared_cell_ : main_graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto send_recv_cut_border = CutBorder(graph);

  std::vector<AnfNodePtr> send_ops;
  (void)(send_ops.insert(send_ops.end(), send_recv_shared_param.first.begin(), send_recv_shared_param.first.end()));
  (void)(send_ops.insert(send_ops.end(), send_recv_cut_border.first.begin(), send_recv_cut_border.first.end()));

  if (IsLastStage() && !enable_share_cell_) {
    return;
  }

  if (!send_ops.empty()) {
    type_ptr_ = send_ops.back()->user_data<Type>(DTYPE);
    shape_ = send_ops.back()->user_data<ValueList>(SHAPE);
  }
  if (!enable_share_cell_) {
    auto make_tuple = CreateMakeTupleNode(main_graph_, send_ops);
    auto zero_outputs = GetZeroOutputs(main_graph_);
    std::vector<AnfNodePtr> out = {NewValueNode(prim::kPrimDepend), zero_outputs, make_tuple};
    auto out_node = main_graph_->NewCNode(out);
    (void)manager_->Replace(main_graph_->output(), out_node);
    return;
  }
  if (!IsLastStage()) {
    HandleGraphOutputs(send_ops);
  }
  std::vector<AnfNodePtr> recv_ops;
  (void)(recv_ops.insert(recv_ops.end(), send_recv_shared_param.second.begin(), send_recv_shared_param.second.end()));
  (void)(recv_ops.insert(recv_ops.end(), send_recv_cut_border.second.begin(), send_recv_cut_border.second.end()));
  HandleGraphInputs(recv_ops);
}

void PipelineTransformer::ElimGraphStage() {
  for (auto &fg : manager_->func_graphs()) {
    fg->set_stage(-1);
    fg->set_segment(-1);
  }
}
void PipelineTransformer::RedundancyNode(const AnfNodePtr &node,
                                         mindspore::HashMap<CNodePtr, std::vector<AnfNodePtr>> *make_tuple_map) {
  auto node_users = manager_->node_users()[node];
  for (auto &node_user_pair : node_users) {
    auto cnode = node_user_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // node->UpdateState, replaced node with U.
    auto fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    if (fg->stage() != -1 && fg != main_graph_) {
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimUpdateState)) {
      auto abs = cnode->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      auto monad_node = NewValueNode(kUMonad);
      if (abs->isa<abstract::AbstractIOMonad>()) {
        monad_node = NewValueNode(kIOMonad);
      }
      manager_->SetEdge(cnode, node_user_pair.second, monad_node);
      continue;
    }
    // node->make_tuple, record with a map, Unified deleted later.
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
      if (make_tuple_map->find(cnode) == (*make_tuple_map).end()) {
        (*make_tuple_map)[cnode] = {node};
      } else {
        (*make_tuple_map)[cnode].push_back(node);
      }
    } else {
      RedundancyNode(node_user_pair.first, make_tuple_map);
    }
  }
}

bool PipelineTransformer::IsRedundancyParameter(const AnfNodePtr &parameter,
                                                const std::vector<AnfNodePtr> &non_cloned_parameters) {
  // RedundancyParameter: other stage's parameters included corresponding cloned parameters.
  auto param_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  if (!param_ptr->has_default()) {
    return false;
  }
  std::set<int64_t> stage_set;
  if (!ParameterIsCloned(parameter)) {
    stage_set = parameter_color_map_.at(parameter);
  } else {
    auto param_name = param_ptr->name();
    auto non_clone_name = param_name.substr(param_name.find_first_of('.') + 1);
    for (auto &param : non_cloned_parameters) {
      auto non_cloned_param = param->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(non_cloned_param);
      if (non_clone_name != non_cloned_param->name()) {
        continue;
      }
      stage_set = parameter_color_map_.at(param);
      break;
    }
  }
  if (stage_set.empty()) {
    return false;
  }
  return stage_set.count(stage_) == 0;
}

bool PipelineTransformer::HasNoUpdateParameter() {
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
    if (requires_grad && stage_set.count(stage_)) {
      return false;
    }
  }
  return true;
}

void PipelineTransformer::FreezeGradient() {
  auto node_users_map = manager_->node_users();
  if (HasNoUpdateParameter() && is_train_) {
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

void PipelineTransformer::ElimParameter() {
  auto parameters = root_->parameters();
  mindspore::HashMap<CNodePtr, std::vector<AnfNodePtr>> make_tuple_map;
  std::vector<AnfNodePtr> cloned_parameters;
  FreezeGradient();
  auto node_users_map = manager_->node_users();
  mindspore::HashMap<std::string, std::vector<AnfNodePtr>> params_map;

  // map {w1_name, {w1}}, and vector<cloned_parameters>
  for (auto &parameter : parameters) {
    if (!ParameterIsCloned(parameter)) {
      auto param_ptr = parameter->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      params_map[param_ptr->name()] = {parameter};
      continue;
    }
    cloned_parameters.push_back(parameter);
  }

  // map {w1_name, {w1, w1_clone_1, w1_clone_2, w1_clone_...}}
  for (auto &param : cloned_parameters) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    auto param_name = param_ptr->name();
    auto non_clone_name = param_name.substr(param_name.find_first_of('.') + 1);
    if (params_map.find(non_clone_name) != params_map.end()) {
      params_map[non_clone_name].push_back(param);
    }
  }
  // check and handle redundancy node
  for (auto &param_pair : params_map) {
    auto params = param_pair.second;
    for (size_t i = 0; i < params.size(); ++i) {
      auto param_ptr = params[i]->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        continue;
      }
      std::set<int64_t> stage_set;
      stage_set = parameter_color_map_.at(params[0]);
      if (stage_set.empty()) {
        continue;
      }
      if (stage_set.count(stage_) == 0) {
        RedundancyNode(params[i], &make_tuple_map);
      }
    }
  }

  for (auto &temp : make_tuple_map) {
    auto make_tuple = temp.first;
    auto fg = make_tuple->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto remove_vector = temp.second;
    if (remove_vector.empty()) {
      continue;
    }
    auto make_tuple_user = node_users_map.at(make_tuple).front().first;
    auto make_tuple_inputs = make_tuple->inputs();
    std::vector<AnfNodePtr> new_inputs;
    for (auto &input : make_tuple_inputs) {
      if (std::find(remove_vector.begin(), remove_vector.end(), input) == remove_vector.end()) {
        new_inputs.push_back(input);
        continue;
      }
      if (root_->has_flag(NO_UPDATE) && IsPrimitiveCNode(make_tuple_user, prim::kPrimAddN)) {
        new_inputs.push_back(CreateZeroseOutput(input, 0));
      }
    }
    auto new_make_tuple = fg->NewCNode(new_inputs);
    (void)manager_->Replace(make_tuple, new_make_tuple);
  }
}

void PipelineTransformer::ModifyParameterList() {
  ElimParameter();
  auto parameters = root_->parameters();
  std::vector<AnfNodePtr> parameter_list;
  for (auto &parameter : parameters) {
    auto param = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (!manager_->node_users()[parameter].empty() || !param->has_default()) {
      parameter_list.push_back(parameter);
    }
  }
  auto del_num = parameters.size() - parameter_list.size();
  root_->set_fv_param_count(root_->fv_param_count() - del_num);
  manager_->SetParameters(root_, parameter_list);
}
}  // namespace parallel
}  // namespace mindspore
