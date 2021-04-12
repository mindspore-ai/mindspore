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

#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "ir/anf.h"
#include "base/core_ops.h"
#include "utils/comm_manager.h"
#include "utils/ms_context.h"
#include "mindspore/core/utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
static std::unordered_map<AnfNodePtr, std::set<int64_t>> parameter_color_map;
// map<rank, tag>
static std::unordered_map<int64_t, int64_t> send_tag_map;
static std::unordered_map<int64_t, int64_t> recv_tag_map;
const std::set<PrimitivePtr> WHITE_LIST = {prim::kPrimCast, prim::kPrimTupleGetItem};

static bool IsInWhiteList(const CNodePtr &cnode) {
  for (auto &prim : WHITE_LIST) {
    if (IsPrimitiveCNode(cnode, prim)) {
      return true;
    }
  }
  return false;
}

static void SetGradTag(const AnfNodePtr &node, const FuncGraphManagerPtr &manager, size_t accum = 0) {
  accum += 1;
  if (accum > MAX_RECURSIVE_DEPTH) {
    return;
  }
  const auto &node_users = manager->node_users()[node];
  for (auto &user_pair : node_users) {
    auto user_node = user_pair.first;
    if (!user_node->grad()) {
      user_node->set_grad(true);
      SetGradTag(user_node, manager, accum);
    }
  }
}

void PipelineTransformer::LabelRequiredGradCNode() {
  auto parameters = root_->parameters();
  for (auto parameter : parameters) {
    if (!ParameterRequireGrad(parameter)) {
      continue;
    }
    SetGradTag(parameter, manager_);
  }
}

void PipelineTransformer::Coloring() {
  auto need_coloring = true;
  std::set<int64_t> stage_set;
  while (need_coloring) {
    need_coloring = false;
    for (auto &fg : manager_->func_graphs()) {
      if (fg == root_) {
        continue;
      }
      auto value_nodes = fg->value_nodes();
      for (auto &value_pair : value_nodes) {
        auto node = value_pair.first;
        if (!IsValueNode<FuncGraph>(node)) {
          continue;
        }
        auto graph = GetValueNode<FuncGraphPtr>(node);
        auto need_grad = graph->get_return()->grad();
        auto node_users = manager_->node_users()[node];
        for (auto &user_pair : node_users) {
          auto user_node = user_pair.first->cast<CNodePtr>();
          user_node->set_stage(graph->stage());
          user_node->set_grad(need_grad);
          auto user_node_graph = user_node->func_graph();
          if (graph->stage() != -1) {
            stage_set.insert(graph->stage());
          }
          if (graph->stage() == stage_ && user_node_graph->stage() == -1) {
            user_node_graph->set_stage(graph->stage());
            need_coloring = true;
          }
        }
      }
    }
  }
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto stage_num = g_device_manager->stage_num();
  if (SizeToLong(stage_set.size()) != stage_num) {
    MS_LOG(EXCEPTION) << "Stage num is " << stage_num << " is not equal to stage used: " << stage_set.size();
  }
  return;
}

void PipelineTransformer::BroadCastColoring() {
  for (auto &fg : manager_->func_graphs()) {
    if (fg == root_ || fg->stage() == -1) {
      continue;
    }
    DoBroadCast(fg);
    SetNoStageNode(fg);
  }
}

bool PipelineTransformer::IsPipelineCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (IsInWhiteList(cnode)) {
    return false;
  }
  if (IsInParallelBlackList(prim)) {
    MS_LOG(INFO) << "PipelineSplit don't care node:" << prim->name();
    return false;
  }
  return true;
}

OperatorInfoPtr PipelineTransformer::CreateOpInfo(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsPipelineCareNode(cnode)) {
    MS_LOG(EXCEPTION) << "Node: " << cnode->ToString() << " is not a Pipeline Care Node.";
  }
  auto shape_list = ExtractShape(cnode);
  if (shape_list.empty()) {
    MS_LOG(EXCEPTION) << "Node: " << cnode->ToString() << " failed to extract shape.";
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == RESHAPE) {
    MS_LOG(EXCEPTION) << "Reshape op can't be a border.";
  }
  auto attrs = prim->attrs();
  auto op_info = OperatorInstance(prim, attrs, shape_list);
  auto &inputs = cnode->inputs();
  std::vector<ValuePtr> input_value;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>()) {
      input_value.push_back(GetValueNode(inputs[index]));
    } else {
      input_value.emplace_back(nullptr);
    }
  }
  op_info->set_input_value(input_value);
  op_info->set_outputs_dtype(cnode->Type());
  op_info->set_cnode(cnode);
  StrategyPtr strategy = nullptr;
  if (!StrategyFound(attrs)) {
    strategy = GenerateBatchParallelStrategy(op_info, prim);
  } else {
    strategy = ExtractStrategy(attrs);
  }
  MS_EXCEPTION_IF_NULL(strategy);
  if (op_info->Init(strategy) == FAILED) {
    MS_LOG(EXCEPTION) << "operator: " << prim->name() << " init failed.";
  }
  return op_info;
}

std::pair<OperatorInfoPtr, TensorInfoPtr> PipelineTransformer::GetOpInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Handle Cast and TupleGetitem situation
  size_t tensor_info_index = 0;
  if (IsPrimitiveCNode(cnode, prim::kPrimCast)) {
    cnode = cnode->input(1)->cast<CNodePtr>();
  } else if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    tensor_info_index = LongToSize(GetTupleGetItemIndex(cnode));
    cnode = cnode->input(1)->cast<CNodePtr>();
  }
  // Create OperatorInfo to get slice_shape for send/recv
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_info = CreateOpInfo(cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  auto tensor_info = op_info->outputs_tensor_info()[tensor_info_index];
  return std::make_pair(op_info, std::make_shared<TensorInfo>(tensor_info));
}

CNodePtr PipelineTransformer::HandleMonadLoad(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto &node_users = manager_->node_users()[node];
  for (auto &user_pair : node_users) {
    auto user_node = user_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_node);
    if (IsPipelineCareNode(user_node)) {
      return user_node;
    }
  }
  return nullptr;
}

std::pair<OperatorInfoPtr, TensorInfoPtr> PipelineTransformer::GetParameterPair(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto &node_users = manager_->node_users()[node];
  for (auto &user_pair : node_users) {
    auto care_node = user_pair.first;
    auto care_cnode = care_node->cast<CNodePtr>();
    if (IsPrimitiveCNode(care_node, prim::kPrimLoad)) {
      care_cnode = HandleMonadLoad(care_node);
      if (!care_cnode) {
        continue;
      }
    } else {
      if (!IsPipelineCareNode(care_cnode)) {
        continue;
      }
    }
    MS_EXCEPTION_IF_NULL(care_cnode);
    auto op_info = CreateOpInfo(care_cnode);
    MS_EXCEPTION_IF_NULL(op_info);
    auto tensor_info = op_info->inputs_tensor_info()[IntToSize(user_pair.second) - 1];
    return std::make_pair(nullptr, std::make_shared<TensorInfo>(tensor_info));
  }
  return std::make_pair(nullptr, nullptr);
}

void PipelineTransformer::DoBroadCast(const FuncGraphPtr &func) {
  auto need_coloring = true;
  while (need_coloring) {
    need_coloring = false;
    auto all_nodes = func->nodes();
    auto &node_users = manager_->node_users();
    for (auto &node : all_nodes) {
      if (node->isa<CNode>() || node->stage() == -1) {
        continue;
      }
      auto stage = node->stage();
      for (auto &user_pair : node_users[node]) {
        auto user_node = user_pair.first->cast<CNodePtr>();
        auto user_node_stage = user_node->stage();
        if (IsValueNode<FuncGraph>(user_node->input(0)) && stage > user_node_stage) {
          user_node->set_stage(stage);
          need_coloring = true;
        }
      }
    }
  }
}

void PipelineTransformer::SetNoStageNode(const FuncGraphPtr &func) {
  auto all_nodes = func->nodes();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>() || node->stage() != -1) {
      continue;
    }
    node->set_stage(0);
  }
}

void PipelineTransformer::HandleSharedParameter() {
  auto parameters = root_->parameters();
  for (auto &parameter : parameters) {
    auto parameter_stage = parameter_color_map[parameter];
    if (parameter_stage.size() <= 1) {
      continue;
    }
    auto users = manager_->node_users()[parameter];
    for (auto &user : users) {
      auto node = user.first;
      auto graph = node->func_graph();
      if (graph != root_ && graph->stage() == -1) {
        MS_LOG(EXCEPTION) << "Don't support this situation.";
      }
      if (graph == root_ || graph->stage() != stage_) {
        continue;
      }
      if (stage_ == *parameter_stage.begin()) {
        std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple)};
        for (auto &stage : parameter_stage) {
          if (stage == stage_) {
            continue;
          } else {
            auto send_out = InsertSend(graph, parameter, stage, stage_);
            make_tuple_input.push_back(send_out.depend);
          }
        }
        auto make_tuple = graph->NewCNode(make_tuple_input);
        OperatorAttrs depend_attrs;
        auto depend_op = CreatOpInstance(depend_attrs, DEPEND, "");
        std::vector<AnfNodePtr> depend_input = {NewValueNode(depend_op), parameter, make_tuple};
        auto depend = graph->NewCNode(depend_input);
        depend->set_abstract(parameter->abstract());
        manager_->SetEdge(node, user.second, depend);
        break;
      } else {
        (void)InsertReceive(graph, parameter, node, user.second, stage_, *parameter_stage.begin());
        break;
      }
    }
  }
}

void PipelineTransformer::ParameterColoring() {
  auto parameters = root_->parameters();
  for (auto &parameter : parameters) {
    auto users = manager_->node_users()[parameter];
    std::set<int64_t> parameter_stage;
    for (auto &user : users) {
      auto node = user.first;
      auto graph = node->func_graph();
      if (graph != root_ && graph->stage() != -1) {
        parameter_stage.insert(graph->stage());
        parameter->set_stage(graph->stage());
      }
    }
    if (*parameter_stage.begin() == stage_ && !virtual_param_) {
      virtual_param_ = parameter;
    }
    parameter_color_map[parameter] = parameter_stage;
  }
}

static std::pair<ValueListPtr, TypePtr> GetShapeType(const AnfNodePtr &node, const Shape &shape) {
  TypePtr type;
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr && IsValueNode<FuncGraph>(cnode->input(0))) {
    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto graph_return = graph->get_return();
    type = graph_return->Type();
  } else {
    type = node->Type();
  }
  MS_EXCEPTION_IF_NULL(type);
  std::vector<ValuePtr> element;
  std::transform(shape.begin(), shape.end(), std::back_inserter(element), [](int elem) { return MakeValue(elem); });
  auto shape_list = std::make_shared<ValueList>(element);
  auto tensor_type = type->cast<mindspore::TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto dtype = tensor_type->element();
  MS_EXCEPTION_IF_NULL(dtype);
  return std::make_pair(shape_list, dtype);
}

AnfNodePtr PipelineTransformer::HandleMonadDepend(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto cnode = node->cast<CNodePtr>();
    return HandleMonadDepend(cnode->input(1));
  }
  return node;
}

AnfNodePtr PipelineTransformer::FindPipelineCareNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto output = HandleMonadDepend(graph->output());
    MS_EXCEPTION_IF_NULL(output);
    if (output->isa<Parameter>()) {
      return output;
    }
    cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
  }
  if (IsInWhiteList(cnode)) {
    return cnode->cast<AnfNodePtr>();
  }
  if (!IsPipelineCareNode(cnode)) {
    MS_LOG(EXCEPTION) << "Only PipelineSplit cared node can be a border.";
  }
  return cnode->cast<AnfNodePtr>();
}

SendAttr PipelineTransformer::InsertSend(const FuncGraphPtr &graph, const AnfNodePtr &parameter,
                                         int64_t user_node_stage, int64_t node_stage) {
  auto dest_rank = global_rank_ + (user_node_stage - node_stage) * per_stage_rank_num_;
  int64_t send_tag;
  if (send_tag_map.find(dest_rank) != send_tag_map.end()) {
    send_tag = send_tag_map[dest_rank] + 1;
    send_tag_map[dest_rank] += 1;
  } else {
    send_tag = 0;
    send_tag_map[dest_rank] = 0;
  }
  Attr attr_tag = std::make_pair("sr_tag", MakeValue(send_tag));
  Attr attr_rank = std::make_pair("dest_rank", MakeValue(dest_rank));
  OperatorAttrs attrs = {attr_tag, attr_rank};
  auto send_op = CreatOpInstance(attrs, SEND, "send");
  auto send_node = NewValueNode(send_op);
  auto prim = GetValueNode<PrimitivePtr>(send_node);
  std::pair<OperatorInfoPtr, TensorInfoPtr> op_info_pair;
  if (parameter->isa<Parameter>()) {
    op_info_pair = GetParameterPair(parameter);
  } else {
    auto care_node = FindPipelineCareNode(parameter);
    if (care_node->isa<Parameter>()) {
      op_info_pair = GetParameterPair(care_node);
    } else {
      op_info_pair = GetOpInfo(care_node);
    }
  }
  auto tensor_info = op_info_pair.second;
  MS_EXCEPTION_IF_NULL(tensor_info);
  auto slice_shape = tensor_info->slice_shape();
  auto shape_type_pair = GetShapeType(parameter, slice_shape);
  prim->set_attr("shape", shape_type_pair.first);
  prim->set_attr("dtype", shape_type_pair.second);
  std::vector<AnfNodePtr> send_input = {send_node, parameter};
  auto send = graph->NewCNode(send_input);
  OperatorAttrs depend_attrs;
  auto depend_op = CreatOpInstance(depend_attrs, DEPEND, "depend");
  std::vector<AnfNodePtr> depend_input = {NewValueNode(depend_op), parameter, send};
  auto depend = graph->NewCNode(depend_input);
  auto abstract = parameter->abstract();
  depend->set_abstract(abstract);
  SendAttr send_out = {shape_type_pair.first, shape_type_pair.second, depend};
  return send_out;
}

AnfNodePtr PipelineTransformer::InsertReceive(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const AnfNodePtr &use_node, int index, int64_t user_node_stage,
                                              int64_t node_stage) {
  auto src_rank = global_rank_ - (user_node_stage - node_stage) * per_stage_rank_num_;
  int64_t recv_tag;
  if (recv_tag_map.find(src_rank) != recv_tag_map.end()) {
    recv_tag = recv_tag_map[src_rank] + 1;
    recv_tag_map[src_rank] += 1;
  } else {
    recv_tag = 0;
    recv_tag_map[src_rank] = 0;
  }
  Attr attr_tag = std::make_pair("sr_tag", MakeValue(recv_tag));
  Attr attr_rank = std::make_pair("src_rank", MakeValue(src_rank));
  std::pair<OperatorInfoPtr, TensorInfoPtr> op_info_pair;
  if (node->isa<Parameter>()) {
    op_info_pair = GetParameterPair(node);
  } else {
    auto care_node = FindPipelineCareNode(node);
    if (care_node->isa<Parameter>()) {
      op_info_pair = GetParameterPair(care_node);
    } else {
      op_info_pair = GetOpInfo(care_node);
    }
  }
  auto tensor_info = op_info_pair.second;
  MS_EXCEPTION_IF_NULL(tensor_info);
  auto slice_shape = tensor_info->slice_shape();
  auto shape_type_pair = GetShapeType(node, slice_shape);
  Attr attr_shape = std::make_pair("shape", shape_type_pair.first);
  Attr attr_dtype = std::make_pair("dtype", shape_type_pair.second);
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_shape, attr_dtype};
  auto recv_op = CreatOpInstance(attrs, RECEIVE, "recv");
  std::vector<AnfNodePtr> recv_input;
  if (node->isa<Parameter>()) {
    recv_input = {NewValueNode(recv_op), node};
  } else {
    if (node->grad()) {
      recv_input = {NewValueNode(recv_op), virtual_param_};
    } else {
      auto param = root_->parameters()[0];
      recv_input = {NewValueNode(recv_op), param};
    }
  }
  auto recv = graph->NewCNode(recv_input);
  auto node_abstract = node->abstract();
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsValueNode<FuncGraph>(cnode->input(0))) {
      auto output = GetValueNode<FuncGraphPtr>(cnode->input(0))->output();
      MS_EXCEPTION_IF_NULL(output);
      node_abstract = output->abstract();
    }
  }
  MS_EXCEPTION_IF_NULL(node_abstract);
  recv->set_abstract(node_abstract);
  if (op_info_pair.first != nullptr) {
    recv->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_info->tensor_layout()));
    recv->set_user_data<OperatorInfo>(op_info_pair.first);
  }
  manager_->SetEdge(use_node, index, recv);
  return recv;
}

bool PipelineTransformer::Reuse(const AnfNodePtr &node, int64_t next_node_stage, int64_t node_stage,
                                const std::vector<AnfNodePtr> &out_input) {
  auto node_users = manager_->node_users()[node];
  auto dest_rank = global_rank_ + (next_node_stage - node_stage) * per_stage_rank_num_;
  for (auto &depend : out_input) {
    if (!IsPrimitiveCNode(depend, prim::kPrimDepend)) {
      continue;
    }
    auto cnode = depend->cast<CNodePtr>();
    if (cnode->input(1) == node) {
      auto send_cnode = cnode->input(2)->cast<CNodePtr>();
      auto prim = GetValueNode<PrimitivePtr>(send_cnode->input(0));
      auto dest_rank_send = GetValue<int64_t>(prim->GetAttr("dest_rank"));
      if (dest_rank_send == dest_rank) {
        return true;
      }
    }
  }
  return false;
}

std::pair<bool, int64_t> PipelineTransformer::IsSharedNode(const AnfNodePtr &node, const AnfNodeIndexSet &node_users) {
  std::set<int64_t> tag_set;
  auto node_stage = node->stage();
  int64_t min_tag = node_stage;
  for (auto &user_pair : node_users) {
    auto user_node = user_pair.first;
    auto user_node_stage = user_node->stage();
    tag_set.insert(user_node_stage);
    if (user_node_stage == -1) {
      continue;
    }
    min_tag = min_tag > user_node_stage ? user_node_stage : min_tag;
  }
  bool is_shared = tag_set.size() > 1;
  return std::make_pair(is_shared, min_tag);
}

void PipelineTransformer::CutBorder(const FuncGraphPtr &graph) {
  OperatorAttrs depend_attrs;
  auto depend_op = CreatOpInstance(depend_attrs, "Depend", "");
  std::vector<AnfNodePtr> out_input = {NewValueNode(depend_op)};
  auto all_nodes = graph->nodes();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>() || node->stage() == -1) {
      continue;
    }
    auto node_users = manager_->node_users()[node];
    auto shared_min_tag_pair = IsSharedNode(node, node_users);
    auto is_shared = shared_min_tag_pair.first;
    auto min_tag = shared_min_tag_pair.second;
    AnfNodePtr receive = nullptr;
    for (auto &user_pair : node_users) {
      auto user_node = user_pair.first;
      auto node_stage = node->stage();
      auto user_node_stage = user_node->stage();
      if (node_stage != stage_ && user_node_stage != stage_) {
        continue;
      }
      if (node_stage < user_node_stage) {
        if (is_shared && (min_tag != node_stage)) {
          continue;
        }
        if (node_stage == stage_) {
          if (Reuse(node, user_node_stage, node_stage, out_input)) {
            continue;
          }
          auto send_out = InsertSend(graph, node, user_node_stage, node_stage);
          out_input.insert(out_input.begin() + 1, send_out.depend);
          type_ptr_ = send_out.type;
          shape_ = send_out.shape;
        } else {
          if (!receive) {
            receive = InsertReceive(graph, node, user_node, user_pair.second, user_node_stage, node_stage);
          } else {
            manager_->SetEdge(user_node, user_pair.second, receive);
          }
        }
        continue;
      }
      if (node_stage > user_node_stage) {
        auto cnode = node->cast<CNodePtr>();
        auto user_cnode = user_node->cast<CNodePtr>();
        if (IsValueNode<FuncGraph>(cnode->input(0)) && IsValueNode<FuncGraph>(user_cnode->input(0))) {
          MS_LOG(EXCEPTION) << "Don't support this situation";
        }
        continue;
      }
    }
  }
  if (out_input.size() == 2) {
    manager_->Replace(graph->output(), out_input[1]);
  }
  if (out_input.size() > 2) {
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    make_tuple_inputs.insert(make_tuple_inputs.begin() + 1, out_input.begin() + 2, out_input.end());
    auto make_tuple = graph->NewCNode(make_tuple_inputs);
    std::vector<AnfNodePtr> out_depend_inputs = {out_input[0], out_input[1], make_tuple};
    auto out_node = graph->NewCNode(out_depend_inputs);
    manager_->Replace(graph->output(), out_node);
  }
}

void PipelineTransformer::CutGraph() {
  for (auto &fg : manager_->func_graphs()) {
    CutBorder(fg);
  }
}

bool PipelineTransformer::IsStageNode(const CNodePtr &node) {
  for (auto &input : node->inputs()) {
    if (input->isa<Parameter>()) {
      return (*parameter_color_map[input].begin() == stage_ || input->stage() == -1);
    } else if (input->isa<CNode>()) {
      auto pre_node = input->cast<CNodePtr>();
      return IsStageNode(pre_node);
    } else {
      continue;
    }
  }
  return true;
}

void PipelineTransformer::ElimGraphStage() {
  for (auto &fg : manager_->func_graphs()) {
    fg->set_stage(-1);
  }
}

std::pair<CNodePtr, FuncGraphPtr> PipelineTransformer::FindSensNode() {
  std::pair<CNodePtr, FuncGraphPtr> sens_graph_pair;
  CNodePtr sens_cnode;
  FuncGraphPtr func_graph;
  for (auto &node : root_->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    sens_cnode = node->cast<CNodePtr>();
    AnfNodePtr expect_tuple_getitem = sens_cnode->input(0);
    MS_EXCEPTION_IF_NULL(expect_tuple_getitem);
    if (!expect_tuple_getitem->isa<CNode>()) {
      continue;
    }

    auto expect_tuple_getitem_cnode = expect_tuple_getitem->cast<CNodePtr>();
    if (!IsPrimitiveCNode(expect_tuple_getitem_cnode, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto expect_anonymous = expect_tuple_getitem_cnode->input(1);
    if (!expect_anonymous->isa<CNode>()) {
      continue;
    }
    auto expect_anonymous_cnode = expect_anonymous->cast<CNodePtr>();
    AnfNodePtr expect_j = expect_anonymous_cnode->input(0);
    if (!expect_j->isa<CNode>()) {
      continue;
    }
    auto expect_j_cnode = expect_j->cast<CNodePtr>();
    if (!IsPrimitiveCNode(expect_j_cnode, prim::kPrimJ)) {
      continue;
    }
    func_graph = GetValueNode<FuncGraphPtr>(expect_j_cnode->input(1));
    break;
  }
  sens_graph_pair = std::make_pair(sens_cnode, func_graph);
  return sens_graph_pair;
}

void PipelineTransformer::CoverSensShape() {
  if (IsLastStage()) {
    return;
  }
  auto sens_graph_pair = FindSensNode();
  auto sens_cnode = sens_graph_pair.first;
  MS_EXCEPTION_IF_NULL(sens_cnode);
  OperatorAttrs attrs;
  auto fill_op = CreatOpInstance(attrs, "Fill", "");
  MS_EXCEPTION_IF_NULL(type_ptr_);
  MS_EXCEPTION_IF_NULL(shape_);
  std::vector<AnfNodePtr> fill_input = {NewValueNode(fill_op), NewValueNode(type_ptr_),
                                        NewValueNode(MakeValue(shape_->value())), NewValueNode(0)};
  auto fill = root_->NewCNode(fill_input);
  std::vector<AnfNodePtr> new_sens_input = {sens_cnode->input(0), fill};
  auto new_sens_node = root_->NewCNode(new_sens_input);
  manager_->Replace(sens_cnode, new_sens_node);
}

void PipelineTransformer::ElimParameter() {
  auto parameters = root_->parameters();
  std::vector<AnfNodePtr> parameter_list;
  for (auto &parameter : parameters) {
    if (!manager_->node_users()[parameter].empty()) {
      parameter_list.push_back(parameter);
    }
  }
  auto del_num = parameters.size() - parameter_list.size();
  root_->set_hyper_param_count(root_->hyper_param_count() - del_num);
  manager_->SetParameters(root_, parameter_list);
}
}  // namespace parallel
}  // namespace mindspore
