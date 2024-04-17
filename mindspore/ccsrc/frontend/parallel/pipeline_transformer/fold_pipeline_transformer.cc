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

#include "frontend/parallel/pipeline_transformer/fold_pipeline_transformer.h"
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/graph_util/graph_splitter.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/parameter_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "ops/other_ops.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
mindspore::HashMap<int64_t, int64_t> fold_send_tag_map;
mindspore::HashMap<int64_t, int64_t> fold_recv_tag_map;

void FoldPipelineTransformer::CreateForwardGroup2() {
  auto rank_id = g_device_manager->global_rank();
  auto stage_id = g_device_manager->stage_id();
  auto stage_num = g_device_manager->stage_num();

  std::vector<int64_t> forward_rank_list;
  forward_rank_list.push_back(rank_id);
  if (stage_id < stage_num - 1) {
    forward_rank_list.push_back(rank_id + per_stage_rank_num_);
  } else {
    forward_rank_list.push_back(rank_id + per_stage_rank_num_ * (0 - stage_id));
  }

  Group g;

  if (g_device_manager->CreateGroup(forward_rank_list, &g) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create forward communication group between all pipeline stages failed, the rank_list is: "
                      << forward_rank_list;
  }

  std::vector<int64_t> backward_rank_list;
  if (stage_id == 0) {
    backward_rank_list.push_back(rank_id + per_stage_rank_num_ * (stage_num - 1));
  } else {
    backward_rank_list.push_back(rank_id - per_stage_rank_num_);
  }
  backward_rank_list.push_back(rank_id);

  Group g_back;
  if (g_device_manager->CreateGroup(backward_rank_list, &g_back) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create backward communication group between all pipeline stages failed, the rank_list is: "
                      << backward_rank_list;
  }

  group_.push_back(g.name());
  group_.push_back(g_back.name());
}
void HandleSegment(const ValuePtr &value, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto nodes = graph->nodes();
  for (auto node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_LOG(INFO) << "Handle Segment cnode: " << cnode->fullname_with_scope();
      cnode->AddPrimalAttr(SEGMENT, value);
    }
  }
}
void FoldPipelineTransformer::Coloring() {
  auto need_coloring = true;
  std::set<int64_t> stage_set;
  std::set<int64_t> segment_set;
  if (!IsTraining(manager_)) {
    is_train_ = false;
  }
  while (need_coloring) {
    need_coloring = false;
    for (auto &fg : manager_->func_graphs()) {
      if (fg == root_ && is_train_) {
        continue;
      }
      auto value_nodes = fg->value_nodes();
      for (auto value_pair = value_nodes.cbegin(); value_pair != value_nodes.cend(); ++value_pair) {
        auto node = (*value_pair).first;
        if (!IsValueNode<FuncGraph>(node)) {
          continue;
        }
        auto graph = GetValueNode<FuncGraphPtr>(node);
        if (graph->stage() == -1) {
          continue;
        }
        (void)stage_set.insert(graph->stage());
        (void)segment_set.insert(graph->segment());
        auto node_users = manager_->node_users()[node];
        HandleSegment(MakeValue(graph->segment()), graph);
        for (auto &user_pair : node_users) {
          auto user_node = user_pair.first->cast<CNodePtr>();
          user_node->set_user_data<NodeStageInfo>(std::make_shared<NodeStageInfo>(graph->stage()));
          user_node->set_user_data<NodeSegmentInfo>(std::make_shared<NodeSegmentInfo>(graph->segment()));
          auto user_node_graph = user_node->func_graph();
          if (graph->stage() == stage_ && user_node_graph->stage() == -1) {
            user_node_graph->set_stage(graph->stage());
            MS_LOG(INFO) << "Set_segment in Coloring" << graph->segment();
            user_node_graph->set_segment(graph->segment());
            need_coloring = true;
          }
        }
      }
    }
  }
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto stage_num = g_device_manager->stage_num();
  auto segment_num = ParallelContext::GetInstance()->pipeline_segment_split_num();
  if (SizeToLong(stage_set.size()) != stage_num) {
    MS_LOG(EXCEPTION) << "Stage num is " << stage_num << " is not equal to stage used: " << stage_set.size();
  }
  if (SizeToLong(segment_set.size()) != segment_num) {
    MS_LOG(EXCEPTION) << "Segment num is " << segment_num << " is not equal to segment used: " << segment_set.size();
  }
}

void FoldPipelineTransformer::ColorForNodes() {
  for (auto &fg : manager_->func_graphs()) {
    auto stage = fg->stage();
    auto segment = fg->segment();
    if (stage < 0) {
      continue;
    }
    if (segment < 0) {
      continue;
    }
    if (fg == root_ || fg == main_graph_ || fg == shared_cell_) {
      continue;
    }
    auto all_nodes = fg->nodes();
    for (auto node : all_nodes) {
      if (node->user_data<NodeStageInfo>() != nullptr) {
        continue;
      }
      node->set_user_data<NodeStageInfo>(std::make_shared<NodeStageInfo>(stage));
      if (node->user_data<NodeSegmentInfo>() != nullptr) {
        continue;
      }
      node->set_user_data<NodeSegmentInfo>(std::make_shared<NodeSegmentInfo>(segment));
    }
  }
}

void FoldPipelineTransformer::BroadCastColoring() {
  auto need_coloring = true;
  while (need_coloring) {
    need_coloring = false;
    auto all_nodes = main_graph_->nodes();
    auto node_users = manager_->node_users();
    for (auto node = all_nodes.cbegin(); node != all_nodes.cend(); ++node) {
      auto stage_info = (*node)->user_data<NodeStageInfo>();
      auto segment_info = (*node)->user_data<NodeSegmentInfo>();
      if (!(*node)->isa<CNode>() || stage_info == nullptr || stage_info->stage() == -1 ||
          IsPrimitiveCNode(*node, prim::kPrimUpdateState)) {
        continue;
      }
      auto stage = stage_info->stage();
      auto segment = segment_info->segment();
      for (auto &user_pair : node_users[*node]) {
        auto user_node = user_pair.first->cast<CNodePtr>();
        auto user_stage_info = user_node->user_data<NodeStageInfo>();
        auto user_segment_info = user_node->user_data<NodeSegmentInfo>();
        if (user_stage_info == nullptr) {
          user_node->set_user_data<NodeStageInfo>(std::make_shared<NodeStageInfo>(stage));
          user_node->set_user_data<NodeSegmentInfo>(std::make_shared<NodeSegmentInfo>(segment));
          need_coloring = true;
          continue;
        }
        auto user_node_stage = user_stage_info->stage();
        auto user_node_segment = user_segment_info->segment();
        if (stage > user_node_stage && segment == user_node_segment) {
          if (IsValueNode<FuncGraph>(user_node->input(0))) {
            MS_LOG(WARNING) << "The stage setting is incorrect. PreNode's stage: " << stage
                            << " is larger than NextNode's stage:" << user_node_stage;
          }
          user_node->set_user_data<NodeStageInfo>(std::make_shared<NodeStageInfo>(stage));
          need_coloring = true;
        }
        if (segment > user_node_segment) {
          user_node->set_user_data<NodeSegmentInfo>(std::make_shared<NodeSegmentInfo>(segment));
          need_coloring = true;
        }
      }
    }
  }
  ColorForNodes();
}

SendAttr FoldPipelineTransformer::InsertSend(const AnfNodePtr &parameter, int64_t user_node_stage, int64_t node_stage,
                                             const ValuePtr &value, int64_t segment) {
  auto dest_rank = global_rank_ + (user_node_stage - node_stage) * per_stage_rank_num_;
  int64_t send_tag;
  auto stage_num = g_device_manager->stage_num();
  if (node_stage == 0 && user_node_stage > 1 && stage_num > 2) {
    if (fold_recv_tag_map.find(dest_rank) != fold_recv_tag_map.end()) {
      send_tag = fold_recv_tag_map[dest_rank] + 1;
      fold_recv_tag_map[dest_rank] += 1;
    } else {
      send_tag = 0;
      fold_recv_tag_map[dest_rank] = 0;
    }
  } else {
    if (fold_send_tag_map.find(dest_rank) != fold_send_tag_map.end()) {
      send_tag = fold_send_tag_map[dest_rank] + 1;
      fold_send_tag_map[dest_rank] += 1;
    } else {
      send_tag = 0;
      fold_send_tag_map[dest_rank] = 0;
    }
  }
  Attr attr_tag = std::make_pair(SR_TAG, MakeValue(send_tag));
  Attr attr_rank = std::make_pair(DEST_RANK, MakeValue(user_node_stage));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_[0]));
  Attr attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[1]));
  if (stage_num > 2) {
    auto next = (user_node_stage == 0) ? 0 : 1;
    attr_rank = std::make_pair(DEST_RANK, MakeValue(next));
    attr_group = std::make_pair(GROUP, MakeValue(group_[0]));
    attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[0]));
  }

  if (node_stage == 0 && user_node_stage > 1 && stage_num > 2) {
    attr_group = std::make_pair(GROUP, MakeValue(group_[1]));
    attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[1]));
    attr_rank = std::make_pair(DEST_RANK, MakeValue(1));
  }
  auto graph = enable_share_cell_ ? shared_cell_ : main_graph_;
  std::vector<AnfNodePtr> send_input = {parameter};
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_group, attr_group_back};
  CNodePtr send = CreateCNodeByInputsAndAttr(graph, SEND, SEND, send_input, attrs);
  auto prim = GetCNodePrimitive(send);
  AnfNodePtr care_node;
  bool is_param = true;
  auto op_info_pair = GetOpInfoPair(parameter, parameter, &care_node, &is_param);
  auto tensor_info = GetTensorInfo(op_info_pair, is_param);

  auto index = op_info_pair.second;
  auto op_info = op_info_pair.first;
  auto slice_shape = tensor_info.slice_shape();
  auto shape_type_pair = GetShapeType(parameter, slice_shape, 0);
  prim->set_attr(SHAPE, shape_type_pair.first);
  prim->set_attr(DTYPE, shape_type_pair.second);
  if (!is_param) {
    send->AddPrimalAttr(PIPELINE_END, value);
  } else {
    send->AddPrimalAttr(PIPELINE_PARAM, value);
    send->set_user_data<OperatorInfo>(op_info);
    send->AddPrimalAttr(PARAM_INDEX, MakeValue(index));
    auto param = care_node ? care_node : parameter;
    send->set_user_data<AnfNode>(INPUT_PARAM, param);
  }
  send->AddPrimalAttr(MICRO, value);
  send->AddPrimalAttr(SEGMENT, MakeValue(segment));
  MS_LOG(INFO) << "Insert Send op, segment is " << segment;
  send->AddPrimalAttr(DEST_RANK, MakeValue(user_node_stage));
  OperatorAttrs depend_attrs;
  CNodePtr depend = CreateCNodeByInputsAndAttr(graph, DEPEND, DEPEND, AnfNodePtrList{parameter, send}, depend_attrs);
  auto abstract = parameter->abstract();
  if (care_node) {
    abstract = care_node->abstract();
  }
  depend->set_abstract(abstract);
  send->set_abstract(abstract);
  SendAttr send_out = {shape_type_pair.first, shape_type_pair.second, depend};

  send->set_user_data<int64_t>(DEST_RANK, std::make_shared<int64_t>(dest_rank));
  send->set_user_data<int64_t>(USER_NODE_STAGE, std::make_shared<int64_t>(user_node_stage));
  return send_out;
}

int64_t FoldPipelineTransformer::ComputeRecvTag(int64_t node_stage, int64_t user_node_stage, int64_t stage_num,
                                                int64_t src_rank) {
  int64_t recv_tag;
  if (node_stage == 0 && user_node_stage > 1 && stage_num > 2) {
    if (fold_send_tag_map.find(src_rank) != fold_send_tag_map.end()) {
      recv_tag = fold_send_tag_map[src_rank] + 1;
      fold_send_tag_map[src_rank] += 1;
    } else {
      recv_tag = 0;
      fold_send_tag_map[src_rank] = 0;
    }
  } else {
    if (fold_recv_tag_map.find(src_rank) != fold_recv_tag_map.end()) {
      recv_tag = fold_recv_tag_map[src_rank] + 1;
      fold_recv_tag_map[src_rank] += 1;
    } else {
      recv_tag = 0;
      fold_recv_tag_map[src_rank] = 0;
    }
  }
  return recv_tag;
}

AnfNodePtr FoldPipelineTransformer::InsertReceive(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const AnfNodePtr &use_node, int index, int64_t user_node_stage,
                                                  int64_t node_stage, const ValuePtr &value,
                                                  const AnfNodePtr &graph_param, int64_t segment) {
  auto src_rank = global_rank_ - (user_node_stage - node_stage) * per_stage_rank_num_;
  auto stage_num = g_device_manager->stage_num();
  auto recv_tag = ComputeRecvTag(node_stage, user_node_stage, stage_num, src_rank);
  Attr attr_tag = std::make_pair(SR_TAG, MakeValue(recv_tag));
  Attr attr_rank = std::make_pair(SRC_RANK, MakeValue(node_stage));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_[0]));
  Attr attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[1]));

  if (stage_num > 2) {
    auto next = (user_node_stage == 0) ? 1 : 0;
    attr_rank = std::make_pair(SRC_RANK, MakeValue(next));
    attr_group = std::make_pair(GROUP, MakeValue(group_[1]));
    attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[1]));
  }
  bool is_param = true;
  AnfNodePtr care_node;
  auto op_info_pair = GetOpInfoPair(node, graph_param, &care_node, &is_param);
  auto tensor_info = GetTensorInfo(op_info_pair, is_param);
  auto tensor_layout = tensor_info.tensor_layout();
  Shape slice_shape = tensor_info.slice_shape();
  auto shape_type_pair = GetShapeType(node, slice_shape, 0);
  Attr attr_shape = std::make_pair(SHAPE, shape_type_pair.first);
  Attr attr_dtype = std::make_pair(DTYPE, shape_type_pair.second);
  if (node_stage == 0 && user_node_stage > 1 && stage_num > 2) {
    attr_group = std::make_pair(GROUP, MakeValue(group_[0]));
    attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[0]));
    attr_rank = std::make_pair(SRC_RANK, MakeValue(0));
  }
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_shape, attr_dtype, attr_group, attr_group_back};
  std::vector<AnfNodePtr> recv_input;
  if (node->isa<Parameter>()) {
    recv_input = {node};
  } else {
    recv_input = {virtual_param_};
  }
  auto recv = CreateCNodeByInputsAndAttr(graph, RECEIVE, RECEIVE, recv_input, attrs);
  if (is_param) {
    recv->set_user_data<AnfNode>(PIPELINE_PARAM, node);
    recv->AddPrimalAttr(PIPELINE_PARAM, value);
    auto param = care_node ? care_node : node;
    recv->set_user_data<AnfNode>(INPUT_PARAM, param);
  } else {
    recv->AddPrimalAttr(PIPELINE_BEGIN, value);
  }
  recv->AddPrimalAttr(MICRO, value);
  recv->AddPrimalAttr(SRC_RANK, MakeValue(node_stage));
  recv->AddPrimalAttr(SEGMENT, MakeValue(segment));
  MS_LOG(INFO) << "Insertreceive segment" << segment;
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
  if (node->isa<Parameter>()) {
    BaseShapePtr parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
    auto abstract_clone = node->abstract()->Clone();
    MS_EXCEPTION_IF_NULL(abstract_clone);
    abstract_clone->set_shape(parallel_shape);
    node->set_abstract(abstract_clone);
    node->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_layout));
    auto actual_param = RefParameterToActualParameter(node);
    if (actual_param) {
      actual_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_layout));
      auto actual_param_abstract = actual_param->abstract()->Clone();
      actual_param_abstract->set_shape(parallel_shape);
      actual_param->set_abstract(actual_param_abstract);
    }
  }
  recv->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_layout));
  recv->set_user_data<OperatorInfo>(op_info_pair.first);

  recv->set_user_data<int64_t>(SRC_RANK, std::make_shared<int64_t>(src_rank));
  recv->set_user_data<int64_t>(NODE_STAGE, std::make_shared<int64_t>(node_stage));
  recv->set_user_data<Type>(SLICE_DTYPE, shape_type_pair.second);
  recv->set_user_data<Shape>(SLICE_SHAPE, std::make_shared<Shape>(slice_shape));

  manager_->SetEdge(use_node, index, recv);
  return recv;
}

AnfNodePtr FoldPipelineTransformer::Reuse(const AnfNodePtr &node, int64_t stage, int64_t node_segment,
                                          const std::vector<AnfNodePtr> &out_input,
                                          const std::vector<int64_t> &out_input_segment, const std::string &tag) {
  std::vector<std::pair<AnfNodePtr, int64_t>> zipped;
  std::transform(out_input.begin(), out_input.end(), out_input_segment.begin(), std::back_inserter(zipped),
                 [](const auto &send, const auto &send_segment) { return std::make_pair(send, send_segment); });

  for (auto &zipp : zipped) {
    auto input = zipp.first;
    auto send_segment = zipp.second;
    auto cnode = input->cast<CNodePtr>();
    if (!cnode) {
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend)) {
      cnode = cnode->input(DEPEND_NODE_SOURCE_INDEX)->cast<CNodePtr>();
    }
    if (cnode->input(1) == node) {
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      auto dest_rank_send = GetValue<int64_t>(prim->GetAttr(tag));
      if (dest_rank_send == stage && node_segment == send_segment) {
        return input;
      }
    }
  }
  return nullptr;
}

AnfNodePtr FoldPipelineTransformer::HandleParameterGraph(const AnfNodePtr &node, const AnfNodePtr &use_node,
                                                         int64_t stage, int64_t user_stage, const ValuePtr &micro,
                                                         size_t pos, const std::vector<AnfNodePtr> &ops) {
  CNodePtr call_node = nullptr;
  auto argument = GetRealKernelNode(node, -1, &call_node).first;

  auto use_cnode = use_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(use_cnode);
  if (!IsValueNode<FuncGraph>(use_cnode->input(0))) {
    MS_LOG(EXCEPTION) << "Parameter must be used by a graph, but got: " << use_cnode->DebugString();
  }
  auto use_graph = GetValueNode<FuncGraphPtr>(use_cnode->input(0));
  auto use_parameter_list = use_graph->parameters();
  auto parameter = use_parameter_list.at(pos - 1);

  // insert receive
  if (stage_ == user_stage) {
    auto recv = PipelineTransformer::Reuse(argument, stage, ops, SRC_RANK);
    if (recv) {
      manager_->SetEdge(use_node, SizeToInt(pos), recv);
      return nullptr;
    }
    auto root_param = argument;
    if (argument->isa<Parameter>() && argument->func_graph() != root_) {
      root_param = GetArgumentsByParameter(argument);
    }
    (void)parameter_color_map_[root_param].insert(user_stage);
    auto graph = enable_share_cell_ ? shared_cell_ : main_graph_;
    return InsertReceive(graph, argument, use_node, SizeToInt(pos), user_stage, stage, micro, parameter, 0);
  }
  // insert send
  if (PipelineTransformer::Reuse(argument, user_stage, ops, DEST_RANK)) {
    return nullptr;
  }
  auto send_out = InsertSend(argument, user_stage, stage_, micro, 0);
  send_out.depend->set_user_data<Type>(DTYPE, send_out.type);
  send_out.depend->set_user_data<ValueList>(SHAPE, send_out.shape);
  return send_out.depend;
}

bool IsStageConflict(int64_t node_stage, int64_t user_node_stage, int64_t node_segment, int64_t user_node_segment,
                     int64_t stage_num, bool isEmbed) {
  if (isEmbed || (node_stage < user_node_stage && node_segment == user_node_segment) ||
      (node_stage == stage_num - 1 && user_node_stage == 0 && node_segment < user_node_segment)) {
    return true;
  }
  return false;
}

void FoldPipelineTransformer::CutBorderForNode(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               std::vector<AnfNodePtr> *send_ops,
                                               std::vector<int64_t> *send_ops_segment,
                                               std::vector<AnfNodePtr> *receive_ops) {
  auto stage_info = node->user_data<NodeStageInfo>();
  auto segment_info = node->user_data<NodeSegmentInfo>();
  auto node_users = manager_->node_users()[node];
  AnfNodePtr receive = nullptr;
  for (auto &user_pair : node_users) {
    auto user_node = user_pair.first;
    auto node_stage = stage_info->stage();
    auto node_segment = segment_info->segment();
    auto user_stage_info = user_node->user_data<NodeStageInfo>();
    if (user_stage_info == nullptr) {
      continue;
    }
    auto user_segment_info = user_node->user_data<NodeSegmentInfo>();
    if (user_segment_info == nullptr) {
      continue;
    }
    auto user_node_stage = user_stage_info->stage();
    if (node_stage != stage_ && user_node_stage != stage_) {
      continue;
    }
    auto micro = user_node->cast<CNodePtr>()->GetPrimalAttr(MICRO);
    auto user_node_segment = user_segment_info->segment();
    if (!micro) {
      MS_LOG(INFO) << "Can't find micro_batch information, use micro(0)";
      micro = MakeValue(int64_t(0));
    }
    auto stage_num = g_device_manager->stage_num();

    bool isEmbed = node_stage < user_node_stage && node_segment != user_node_segment;
    if (IsStageConflict(node_stage, user_node_stage, node_segment, user_node_segment, stage_num, isEmbed)) {
      if (node_stage == stage_) {
        if (IsParameterGraph(node) && isEmbed) {
          auto send_depend = HandleParameterGraph(node, user_node, node_stage, user_node_stage, micro,
                                                  IntToSize(user_pair.second), *send_ops);
          if (!send_depend) {
            continue;
          }
          (void)send_ops->insert(send_ops->cbegin(), send_depend);
          (void)send_ops_segment->insert(send_ops_segment->begin(), node_segment);
          continue;
        }
        if (Reuse(node, user_node_stage, user_node_segment, *send_ops, *send_ops_segment, DEST_RANK)) {
          continue;
        }
        auto send_out = InsertSend(node, user_node_stage, node_stage, micro, node_segment);
        MS_EXCEPTION_IF_NULL(send_out.depend);
        send_ops->push_back(send_out.depend);
        send_ops_segment->push_back(node_segment);
        send_out.depend->set_user_data<Type>(DTYPE, send_out.type);
        send_out.depend->set_user_data<ValueList>(SHAPE, send_out.shape);
      } else {
        if (!receive) {
          if (IsParameterGraph(node)) {
            receive = HandleParameterGraph(node, user_node, node_stage, user_node_stage, micro,
                                           IntToSize(user_pair.second), *receive_ops);
            if (!receive) {
              continue;
            }
            receive_ops->push_back(receive);
          } else {
            receive = InsertReceive(graph, node, user_node, user_pair.second, user_node_stage, node_stage, micro, node,
                                    user_node_segment);
            receive_ops->push_back(receive);
          }
        } else {
          manager_->SetEdge(user_node, user_pair.second, receive);
        }
      }
      continue;
    }
    if (node_stage > user_node_stage && node_segment == user_node_segment) {
      MS_LOG(EXCEPTION) << "Within a segment, node_stage: " << node_stage
                        << " must be smaller than user_node_stage: " << user_node_stage;
    }
  }
}

std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> FoldPipelineTransformer::CutBorder(
  const FuncGraphPtr &graph) {
  std::vector<AnfNodePtr> send_ops;
  std::vector<int64_t> send_ops_segment;
  std::vector<AnfNodePtr> receive_ops;
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::reverse(all_nodes.begin(), all_nodes.end());
  auto stage_num = g_device_manager->stage_num();
  if (is_train_ && (stage_num > micro_size_)) {
    MS_LOG(EXCEPTION) << "MicroBatch size: " << micro_size_ << " can't less than stage num: " << stage_num;
  }
  for (auto &node : all_nodes) {
    auto stage_info = node->user_data<NodeStageInfo>();
    if (!node->isa<CNode>() || stage_info == nullptr || stage_info->stage() == -1 ||
        IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      continue;
    }
    CutBorderForNode(graph, node, &send_ops, &send_ops_segment, &receive_ops);
  }
  RemoveMonadNode();
  return std::make_pair(send_ops, receive_ops);
}

std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> FoldPipelineTransformer::HandleSharedParameter() {
  auto parameters = root_->parameters();
  std::vector<AnfNodePtr> sends = {};
  std::vector<AnfNodePtr> recvs = {};
  for (auto &parameter : parameters) {
    auto parameter_stage = parameter_color_map_[parameter];
    if (parameter_stage.size() <= 1) {
      continue;
    }
    const auto &node_users_map = manager_->node_users();
    auto users = GetParameterLoadUsers(parameter, node_users_map);
    for (auto &user : users) {
      auto node = user.first;
      auto cnode = node->cast<CNodePtr>();
      auto graph = node->func_graph();
      if (IsValueNode<FuncGraph>(cnode->input(0))) {
        graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
      }
      if (graph == root_ || graph->stage() == -1 || parameter_stage.count(stage_) == 0) {
        continue;
      }
      auto micro = cnode->GetPrimalAttr(MICRO);
      if (!micro) {
        MS_LOG(INFO) << "Parameter: " << parameter->ToString() << " doesn't have micro batch";
        micro = MakeValue(int64_t(0));
      }
      if (stage_ == *parameter_stage.begin()) {
        auto user_stage = graph->stage();
        auto stage_info = node->user_data<NodeStageInfo>();
        if (stage_info) {
          user_stage = stage_info->stage();
        }
        if (graph->stage() == stage_ || user_stage == -1) {
          continue;
        }
        if (PipelineTransformer::Reuse(parameter, user_stage, sends, DEST_RANK)) {
          continue;
        }
        auto send_out = InsertSend(parameter, user_stage, stage_, micro, 0);
        sends.push_back(send_out.depend);
      } else {
        auto receive = PipelineTransformer::Reuse(parameter, *parameter_stage.begin(), recvs, SRC_RANK);
        if (receive) {
          manager_->SetEdge(node, user.second, receive);
        } else {
          AnfNodePtr recv;
          auto fg = enable_share_cell_ ? shared_cell_ : main_graph_;
          recv = InsertReceive(fg, parameter, node, user.second, stage_, *parameter_stage.begin(), micro, parameter, 0);
          (void)(recvs.push_back(recv));
        }
      }
    }
  }
  return std::make_pair(sends, recvs);
}

void FoldPipelineTransformer::CutGraph() {
  CreateForwardGroup2();
  MS_EXCEPTION_IF_NULL(main_graph_);
  auto send_recv_shared_param = HandleSharedParameter();
  auto graph = enable_share_cell_ ? shared_cell_ : main_graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto send_recv_cut_border = CutBorder(graph);
  std::vector<AnfNodePtr> send_ops;
  (void)(send_ops.insert(send_ops.end(), send_recv_shared_param.first.begin(), send_recv_shared_param.first.end()));
  (void)(send_ops.insert(send_ops.end(), send_recv_cut_border.first.begin(), send_recv_cut_border.first.end()));
  if (IsLastStage() && !enable_share_cell_) {
    auto out_node = main_graph_->output();

    auto make_tuple = CreateMakeTupleNode(main_graph_, send_ops);

    std::vector<AnfNodePtr> tuple_out_depend = {NewValueNode(prim::kPrimDepend)};
    tuple_out_depend.push_back(out_node);
    tuple_out_depend.push_back(make_tuple);

    auto tuple_out_depend_node = main_graph_->NewCNode(tuple_out_depend);
    tuple_out_depend_node->set_abstract(out_node->abstract());
    (void)manager_->Replace(main_graph_->output(), tuple_out_depend_node);
    return;
  }
  if (send_ops.empty() && !is_train_) {
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
  fold_send_tag_map.clear();
  fold_recv_tag_map.clear();
  if (!IsLastStage()) {
    HandleGraphOutputs(send_ops);
  }
  std::vector<AnfNodePtr> recv_ops;
  (void)(recv_ops.insert(recv_ops.end(), send_recv_shared_param.second.begin(), send_recv_shared_param.second.end()));
  (void)(recv_ops.insert(recv_ops.end(), send_recv_cut_border.second.begin(), send_recv_cut_border.second.end()));
  HandleGraphInputs(recv_ops);
}

}  // namespace parallel
}  // namespace mindspore
