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
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/parameter_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_splitter.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "ir/func_graph_cloner.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"
#include "mindspore/core/utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
static AbstractBasePtr GetRealAbstract(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto &input = node->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(input);
    return input->abstract();
  }
  return node->abstract();
}

bool PipelineInterleave::MainGraph() {
  bool find_main_graph = false;
  for (auto &fg : manager_->func_graphs()) {
    for (auto &node : fg->nodes()) {
      if (IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
        main_graph_ = fg;
        main_graph_->set_flag(MAIN_GRAPH, true);
        virtual_dataset_ = node;
        find_main_graph = true;
        break;
      }
    }
    if (find_main_graph) {
      break;
    }
  }
  if (!find_main_graph) {
    MS_LOG(WARNING) << "Can't find main graph, possible reason is can't find virtual dataset.";
    return false;
  }
  auto value_nodes = main_graph_->value_nodes();
  for (auto value_pair = value_nodes.cbegin(); value_pair != value_nodes.cend(); ++value_pair) {
    auto node = (*value_pair).first;
    if (!IsValueNode<FuncGraph>(node)) {
      continue;
    }
    auto graph = GetValueNode<FuncGraphPtr>(node);
    MS_EXCEPTION_IF_NULL(graph);
    if (!graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      continue;
    }
    shared_cell_ = graph;
    break;
  }
  if (!shared_cell_) {
    MS_LOG(ERROR) << "Pipeline parallel now only support shared_cell.";
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    auto is_pp_interleave = parallel_context->pipeline_interleave();
    if (is_pp_interleave) {
      MS_LOG(EXCEPTION) << "Using pipeline parallel with interleave, should enable lazy_inline.";
    }
    return false;
  }
  return true;
}

void PipelineInterleave::CreateSendReceiveGroup() {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto rank_list = g_device_manager->GetDeviceListBetweenStage();
  auto dev_list = g_device_manager->CreateDeviceListByRankList(rank_list);
  Group forward_send_group;
  if (g_device_manager->CreateGroup(rank_list, &forward_send_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create forward Send communication group failed, the rank list is: " << rank_list;
  }
  group_.emplace_back(forward_send_group.name());

  Group backward_send_group;
  auto backward_send_group_name = forward_send_group.name() + BACKWARD;
  if (g_device_manager->CreateGroup(backward_send_group_name, dev_list, &backward_send_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create backward Send communication group failed, the rank list is: " << rank_list;
  }
  group_.emplace_back(backward_send_group_name);

  Group forward_recv_group;
  auto forward_recv_group_name = forward_send_group.name() + RECEIVE;
  if (g_device_manager->CreateGroup(forward_recv_group_name, dev_list, &forward_recv_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create forward Receive communication group failed, the rank list is: " << rank_list;
  }
  group_.emplace_back(forward_recv_group_name);

  Group backward_recv_group;
  auto backward_recv_group_name = forward_recv_group_name + BACKWARD;
  if (g_device_manager->CreateGroup(backward_recv_group_name, dev_list, &backward_recv_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create backward Receive communication group failed, the rank list is: " << rank_list;
  }
  group_.emplace_back(backward_recv_group_name);
}

ValuePtr PipelineInterleave::SetMicroBatch(const AnfNodePtr &node, int64_t micro_size, size_t batch_axis) const {
  if (!IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
    MS_LOG(EXCEPTION) << "Can't find MicroBatch information.";
  }
  auto cnode = node->cast<CNodePtr>();

  int64_t micro = 0;
  auto value = GetValueNode(cnode->input(2));
  if (value != nullptr) {
    auto tuple = GetValue<std::vector<int64_t>>(value);  // begin
    auto input_tmp = GetNodeShape(cnode->input(1));
    auto input_shape = input_tmp.at(0);
    auto slice_batch_size = input_shape.at(batch_axis);  // betch shape
    if (slice_batch_size == 0) {
      MS_LOG(EXCEPTION) << "slice_batch_size should be a positive integer, but got " << slice_batch_size;
    }
    micro = tuple.at(batch_axis) * micro_size / slice_batch_size;  // micro-index
  } else {
    // dynamic shape
    // if micro is not 1: stridedslice --> maketuple --> scalarmul --> micro
    // if micro is 1: stridedslice --> maketuple --> scalarfloordiv
    if (!IsPrimitiveCNode(cnode->input(2), prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION) << "The begin of stridedslice is not constant value, and not make tuple";
    }
    auto make_tuple_cnode = cnode->input(2)->cast<CNodePtr>();
    if (IsPrimitiveCNode(make_tuple_cnode->input(1), prim::kPrimScalarMul)) {
      auto scalar_mul_cnode = make_tuple_cnode->input(1)->cast<CNodePtr>();
      auto mul_value = GetValueNode(scalar_mul_cnode->input(2));
      micro = GetValue<int64_t>(mul_value);
    } else if (IsPrimitiveCNode(make_tuple_cnode->input(1), prim::kPrimScalarFloorDiv)) {
      micro = 1;
    } else {
      MS_LOG(EXCEPTION) << "Can not find the micro info, the input op of make tuple is "
                        << GetCNodePrimitive(make_tuple_cnode->input(1))->name();
    }
  }

  cnode->AddPrimalAttr(MICRO, MakeValue(micro));
  cnode->AddPrimalAttr(PIPELINE_BEGIN, MakeValue(micro));
  int64_t seg = 0;
  cnode->AddPrimalAttr(SEGMENT, MakeValue(seg));
  return MakeValue(micro);
}

void PipelineInterleave::Init() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  world_group_ = GetWorldGroup();
  uint32_t world_rank_size = 0;
  global_rank_ = parallel::ParallelContext::GetInstance()->global_rank();
  uint32_t rank_id = 0;
  if (!parallel::ParallelContext::GetInstance()->global_rank_is_set()) {
    if (!CommManager::GetInstance().GetRankID(world_group_, &rank_id)) {
      MS_LOG(EXCEPTION) << "Get rank id failed.";
    }
    global_rank_ = UintToInt(rank_id);
  }
  int64_t device_num = 0;
  auto stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (!parallel::ParallelContext::GetInstance()->device_num_is_set()) {
    if (!CommManager::GetInstance().GetRankSize(world_group_, &world_rank_size)) {
      MS_LOG(EXCEPTION) << "Get rank size failed";
    }
    device_num = UintToInt(world_rank_size);
    MS_LOG(INFO) << "Get device num from communication model, the device num is  " << device_num;
  } else {
    device_num = parallel::ParallelContext::GetInstance()->device_num();
  }
  per_stage_rank_num_ = device_num / stage_num;
  return;
}

size_t PipelineInterleave::GetBatchAxisForInput(const AnfNodeIndexSet &input_node_users) const {
  Shapes inputs_tuple;
  for (const auto &input_node_user : input_node_users) {
    auto node = input_node_user.first;
    if (!IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
      return 0;  // simply return 0 when dynamic shape
    }
    auto cnode = node->cast<CNodePtr>();
    auto value = GetValueNode(cnode->input(2));
    if (value == nullptr) {
      return 0;  // simply return 0 when dynamic shape
    }
    auto tuple = GetValue<std::vector<int64_t>>(value);
    inputs_tuple.push_back(tuple);
  }
  size_t batch_axis = 0;
  size_t batch_axis_count = 0;
  size_t input_dim = inputs_tuple.at(0).size();
  size_t micro_num = inputs_tuple.size();
  for (size_t axis = 0; axis < input_dim; ++axis) {
    for (size_t i = 1; i < micro_num; ++i) {
      if (inputs_tuple[i][axis] != inputs_tuple[i - 1][axis]) {
        batch_axis = axis;
        ++batch_axis_count;
        break;
      }
    }
  }
  if (batch_axis_count != kSizeOne) {
    MS_LOG(EXCEPTION)
      << "For pipeline parallelism, micro_size partitioning of the input along a certain dimension is and "
      << "is only allowed, but it is found that " << batch_axis_count << " to be partitioned.";
  }
  return batch_axis;
}

void PipelineInterleave::LabelMicroBatch() {
  if (!is_train_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(virtual_dataset_);
  auto node_user_map = manager_->node_users();
  auto node_users = node_user_map[virtual_dataset_];
  for (auto &node_user : node_users) {
    if (IsPrimitiveCNode(node_user.first, prim::kPrimTupleGetItem)) {
      auto data_users = manager_->node_users()[node_user.first];
      auto node_first = data_users.front().first;
      if (!IsPrimitiveCNode(node_first, prim::kPrimStridedSlice) && !IsPrimitiveCNode(node_first, prim::kPrimShape)) {
        data_users.clear();
        data_users = node_user_map[node_first];
      }
      auto micro_size = int64_t(MicroSize(data_users));
      micro_size_ = micro_size;
      auto batch_axis = GetBatchAxisForInput(data_users);
      MS_LOG(INFO) << "For the "
                   << GetSerialNumberString(
                        GetValue<int64_t>(GetValueNode(node_user.first->cast<CNodePtr>()->input(kIndex2))))
                   << "input, batch axis is " << batch_axis << ", micro size is : " << micro_size;
      for (auto &data_user : data_users) {
        if (!IsPrimitiveCNode(data_user.first, prim::kPrimStridedSlice)) {
          continue;
        }
        auto micro = SetMicroBatch(data_user.first, micro_size, batch_axis);
        SetStridedSliceStrategy(data_user.first);
        auto cnode = data_user.first->cast<CNodePtr>();
        BroadCastMicroBatch(cnode, &node_user_map, micro, 0);
      }
    }
  }
}

void PipelineInterleave::LabelGenMaskFusion() {
  auto fgs = manager_->func_graphs();
  int64_t fusion_id = 0;
  for (auto fg = fgs.cbegin(); fg != fgs.cend(); ++fg) {
    if (*fg == root_ || *fg == main_graph_) {
      continue;
    }
    auto stage = (*fg)->stage();
    if (stage != -1 && stage != stage_) {
      continue;
    }
    auto nodes = (*fg)->nodes();
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {
      if (!IsPrimitiveCNode(*node, prim::kPrimDropoutGenMask) && !IsPrimitiveCNode(*node, prim::kPrimDropoutDoMaskV3) &&
          !IsPrimitiveCNode(*node, prim::kPrimDropout)) {
        continue;
      }
      auto cnode = (*node)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->AddPrimalAttr(kAttrFusion, MakeValue(fusion_id));
      fusion_id += 1;
    }
  }
}

void PipelineInterleave::Coloring() {
  auto need_coloring = true;
  std::set<int64_t> stage_set;
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
        auto node_users = manager_->node_users()[node];
        for (auto &user_pair : node_users) {
          auto user_node = user_pair.first->cast<CNodePtr>();
          user_node->set_user_data<NodeStageInfo>(std::make_shared<NodeStageInfo>(graph->stage()));
          auto user_node_graph = user_node->func_graph();
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
    MS_LOG(EXCEPTION) << "Stage num is " << stage_num << " which is not equal to stage used: " << stage_set.size();
  }
}

void PipelineInterleave::BroadCastColoring() {
  auto need_coloring = true;
  while (need_coloring) {
    need_coloring = false;
    auto all_nodes = shared_cell_->nodes();
    auto node_users = manager_->node_users();
    for (auto node = all_nodes.cbegin(); node != all_nodes.cend(); ++node) {
      auto stage_info = (*node)->user_data<NodeStageInfo>();
      if (!(*node)->isa<CNode>() || stage_info == nullptr || stage_info->stage() == -1 ||
          IsPrimitiveCNode(*node, prim::kPrimUpdateState)) {
        continue;
      }
      auto cnode = (*node)->cast<CNodePtr>();
      auto stage = stage_info->stage();
      auto chunk = stage_info->chunk();
      for (auto &user_pair : node_users[*node]) {
        auto user_node = user_pair.first->cast<CNodePtr>();
        auto user_stage_info = user_node->user_data<NodeStageInfo>();
        if (user_stage_info == nullptr) {
          user_node->set_user_data<NodeStageInfo>(std::make_shared<NodeStageInfo>(stage, chunk));
          need_coloring = true;
          user_node->AddPrimalAttr(CHUNK, MakeValue(chunk));
          user_node->AddPrimalAttr(STAGE, MakeValue(stage));
          continue;
        }
        auto user_node_stage = user_stage_info->stage();
        auto user_node_chunk = user_stage_info->chunk();
        if (stage == user_node_stage) {
          if (chunk > user_node_chunk) {
            user_stage_info->set_chunk(chunk);
            need_coloring = true;
            user_node->AddPrimalAttr(CHUNK, MakeValue(chunk));
            user_node->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
            continue;
          }
          if (chunk < user_node_chunk) {
            stage_info->set_chunk(user_node_chunk);
            chunk = user_node_chunk;
            need_coloring = true;
            cnode->AddPrimalAttr(CHUNK, MakeValue(chunk));
            cnode->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
            continue;
          }
        }
        if (stage > user_node_stage) {
          if ((chunk >= user_node_chunk)) {
            user_stage_info->set_chunk(chunk + 1);
            need_coloring = true;
            user_node->AddPrimalAttr(CHUNK, MakeValue(chunk + 1));
            user_node->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
            continue;
          }
        }
        if ((stage < user_node_stage) && (chunk > user_node_chunk)) {
          user_stage_info->set_chunk(chunk);
          need_coloring = true;
          user_node->AddPrimalAttr(CHUNK, MakeValue(chunk));
          user_node->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
        }
      }
    }
  }
}

std::vector<AnfNodePtr> PipelineInterleave::GetLoadNodeByParam(const AnfNodePtr &param) const {
  std::vector<AnfNodePtr> load_vec = {param};
  auto node_users = manager_->node_users()[param];
  for (auto &param_user : node_users) {
    if (IsPrimitiveCNode(param_user.first, prim::kPrimLoad)) {
      auto graph = param_user.first->func_graph();
      // exclude opt graphs
      if (graph == root_ || (graph->stage() == -1 && graph != main_graph_)) {
        continue;
      }
      (void)load_vec.emplace_back(param_user.first);
    }
  }
  return load_vec;
}

bool PipelineInterleave::GetStageByArgument(const CNodePtr &node, size_t index,
                                            const std::vector<AnfNodePtr> &parameters,
                                            const NodeUsersMap &node_users_map,
                                            std::set<int64_t> *const parameter_stage) {
  if (index < 1) {
    return false;
  }
  const auto &input = node->input(0);
  if (!IsValueNode<FuncGraph>(input)) {
    return false;
  }
  if (GetValueNode<FuncGraphPtr>(input) != shared_cell_) {
    return false;
  }
  auto pos = index - 1;
  const auto &param = parameters.at(pos);
  MS_EXCEPTION_IF_NULL(param);
  auto loads = GetLoadNodeByParam(param);
  const auto &iter = node_users_map.find(loads.back());
  if (iter == node_users_map.end()) {
    return true;
  }
  const auto &users = (*iter).second;
  for (auto &user : users) {
    auto user_cnode = user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    auto stage_info = user_cnode->user_data<NodeStageInfo>();
    if (stage_info != nullptr && stage_info->stage() != -1) {
      (void)((*parameter_stage).insert(stage_info->stage()));
    } else {
      auto graph = user_cnode->func_graph();
      MS_EXCEPTION_IF_NULL(graph);
      if (graph != root_ && graph != main_graph_ && graph != shared_cell_ && graph->stage() != -1) {
        (void)((*parameter_stage).insert(graph->stage()));
      }
    }
  }
  return true;
}

void PipelineInterleave::ParameterColoring() {
  auto parameters = root_->parameters();
  auto &node_users_map = manager_->node_users();
  const auto &share_cell_parameters = shared_cell_->parameters();
  for (auto &parameter : parameters) {
    auto loads = GetLoadNodeByParam(parameter);
    std::set<int64_t> parameter_stage;
    for (auto &load : loads) {
      auto load_users = node_users_map[load];
      for (auto &load_user : load_users) {
        auto user_cnode = load_user.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(user_cnode);
        if (GetStageByArgument(user_cnode, load_user.second, share_cell_parameters, node_users_map, &parameter_stage)) {
          continue;
        }
        auto stage_info = user_cnode->user_data<NodeStageInfo>();
        if (stage_info != nullptr && stage_info->stage() != -1) {
          (void)parameter_stage.insert(stage_info->stage());
          continue;
        } else {
          auto graph = user_cnode->func_graph();
          MS_EXCEPTION_IF_NULL(graph);
          if (graph != root_ && graph != main_graph_ && graph != shared_cell_ && graph->stage() != -1) {
            (void)parameter_stage.insert(graph->stage());
            continue;
          }
        }
      }
    }
    parameter_color_map_[parameter] = parameter_stage;
  }
}

void PipelineInterleave::RemoveMonadNode() {
  auto all_nodes = DeepScopedGraphSearch(shared_cell_->get_return());
  auto node_users_map = manager_->node_users();
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto abs = cnode->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto stage_info = cnode->user_data<NodeStageInfo>();
    if (stage_info == nullptr) {
      continue;
    }
    auto stage = stage_info->stage();
    if (stage != stage_ && stage != -1) {
      auto node_users = node_users_map[node];
      for (auto &user_node : node_users) {
        auto monad_node = NewValueNode(kUMonad);
        if (abs->isa<abstract::AbstractIOMonad>()) {
          monad_node = NewValueNode(kIOMonad);
        }
        manager_->SetEdge(user_node.first, user_node.second, monad_node);
      }
    }
  }
}

static tensor::TensorPtr CreateZeroseOutput(const AnfNodePtr &node, size_t index) {
  auto out_shapes = GetNodeShape(node);
  auto out_shape_type = GetShapeType(node, out_shapes.at(index), index);
  auto zero_tensor = TensorConstructUtils::CreateZerosTensor(out_shape_type.second, out_shapes.at(index));
  return zero_tensor;
}

static AnfNodePtr CreateTupleZeroTensor(const FuncGraphPtr &graph, const AnfNodePtr &node, size_t index) {
  std::vector<AnfNodePtr> temp_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  auto out_shapes = GetNodeShape(node);
  for (size_t ele = 0; ele < out_shapes.size(); ++ele) {
    temp_tuple_inputs.emplace_back(NewValueNode(CreateZeroseOutput(node, ele)));
  }
  auto temp_tuple = graph->NewCNode(temp_tuple_inputs);
  return temp_tuple;
}

void PipelineInterleave::InsertSendReceive(const AnfNodePtr &node, const AnfNodePtr &user_node, int64_t order) {
  auto node_stage_info = node->user_data<NodeStageInfo>();
  auto user_node_stage_info = user_node->user_data<NodeStageInfo>();
  auto node_stage = node_stage_info->stage();
  auto user_stage = user_node_stage_info->stage();
  Attr attr_tag = std::make_pair(SR_TAG, MakeValue(0));
  Attr attr_rank = std::make_pair(DEST_RANK, MakeValue(user_stage));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_[0]));
  Attr attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[1]));
  if (node_stage > user_stage) {
    attr_group = std::make_pair(GROUP, MakeValue(group_[INDEX_TWO]));
    attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_[INDEX_THREE]));
  }
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_group, attr_group_back};
  auto send_op = CreateOpInstance(attrs, SEND, SEND);
  auto send_node = NewValueNode(send_op);
  std::vector<AnfNodePtr> send_input = {send_node, node};
  auto graph = shared_cell_;
  auto send = graph->NewCNode(send_input);
  send->set_user_data<NodeStageInfo>(node_stage_info);
  send->set_abstract(node->abstract());
  send->AddPrimalAttr(CHUNK, MakeValue(node_stage_info->chunk()));
  send->AddPrimalAttr(STAGE, MakeValue(node_stage_info->stage()));
  send->AddPrimalAttr(ORDER, MakeValue(order));

  attr_rank = std::make_pair(SRC_RANK, MakeValue(node_stage));
  auto shape_type_pair = GetShapeType(node, {1}, 0);
  Attr attr_shape = std::make_pair(SHAPE, shape_type_pair.first);
  Attr attr_dtype = std::make_pair(DTYPE, shape_type_pair.second);
  auto send_prim = GetCNodePrimitive(send);
  send_prim->set_attr(DTYPE, shape_type_pair.second);
  OperatorAttrs attrs_recv = {attr_tag, attr_rank, attr_shape, attr_dtype, attr_group, attr_group_back};
  auto recv_op = CreateOpInstance(attrs_recv, RECEIVE, RECEIVE);
  std::vector<AnfNodePtr> recv_input = {NewValueNode(recv_op), send};
  auto recv = graph->NewCNode(recv_input);
  recv->set_abstract(node->abstract());
  recv->set_user_data<NodeStageInfo>(user_node_stage_info);
  recv->AddPrimalAttr(CHUNK, MakeValue(user_node_stage_info->chunk()));
  recv->AddPrimalAttr(STAGE, MakeValue(user_node_stage_info->stage()));
  recv->AddPrimalAttr(ORDER, MakeValue(order));
  auto micro = user_node->cast<CNodePtr>()->GetPrimalAttr(MICRO);
  if (micro != nullptr) {
    recv->AddPrimalAttr(MICRO, micro);
  }
  manager_->Replace(node, recv);
}

void PipelineInterleave::CutBorderForNode(const FuncGraphPtr &graph, const AnfNodePtr &node, int64_t *order) {
  auto stage_info = node->user_data<NodeStageInfo>();
  auto node_users = manager_->node_users()[node];
  AnfNodePtr receive = nullptr;
  auto pre_node = GetRealKernelNode(node, -1).first;
  bool send_param = false;
  if (pre_node->isa<Parameter>()) {
    send_param = true;
  }
  for (auto &user_pair : node_users) {
    auto user_node = user_pair.first;
    auto node_stage = stage_info->stage();
    auto user_stage_info = user_node->user_data<NodeStageInfo>();
    if (user_stage_info == nullptr) {
      continue;
    }
    auto user_node_stage = user_stage_info->stage();
    auto micro = user_node->cast<CNodePtr>()->GetPrimalAttr(MICRO);
    if (!micro) {
      MS_LOG(INFO) << "Can't find micro_batch information, use micro(0)";
      micro = MakeValue(int64_t(0));
    }
    if (node_stage != user_node_stage) {
      InsertSendReceive(node, user_node, *order);
      (*order) += 1;
      if (send_param) {
        parameter_color_map_[pre_node].insert(user_node_stage);
      }
    }
  }
}

void PipelineInterleave::RedundancyNode(const AnfNodePtr &node,
                                        mindspore::HashMap<CNodePtr, std::vector<AnfNodePtr>> *make_tuple_map) {
  auto node_users = manager_->node_users()[node];
  for (auto &node_user_pair : node_users) {
    auto cnode = node_user_pair.first->cast<CNodePtr>();
    // node->UpdateState, replaced node wiht U.
    auto fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    if (fg->stage() != -1 && fg != main_graph_) {
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimUpdateState)) {
      auto u_node = NewValueNode(kUMonad);
      manager_->SetEdge(cnode, node_user_pair.second, u_node);
      continue;
    }
    // node->make_tuple, record with a map, Unified deleted later.
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      if (fg == main_graph_) {
        continue;
      }
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

bool PipelineInterleave::IsRedundancyParameter(const AnfNodePtr &parameter,
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
    auto parameters = root_->parameters();
    auto param_name = param_ptr->name();
    auto non_clone_name = param_name.substr(param_name.find_first_of('.') + 1);
    for (auto &param : non_cloned_parameters) {
      auto non_cloned_param = param->cast<ParameterPtr>();
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

void PipelineInterleave::ElimParameter() {
  auto parameters = root_->parameters();
  mindspore::HashMap<CNodePtr, std::vector<AnfNodePtr>> make_tuple_map;
  std::vector<AnfNodePtr> non_cloned_parameters;
  FreezeGradient();
  auto node_users_map = manager_->node_users();
  for (auto &parameter : parameters) {
    if (ParameterIsCloned(parameter)) {
      continue;
    }
    non_cloned_parameters.push_back(parameter);
  }
  for (auto &parameter : parameters) {
    if (!IsRedundancyParameter(parameter, non_cloned_parameters)) {
      continue;
    }
    MS_LOG(INFO) << "Parameter:" << parameter->DebugString() << " is Redundancy.";
    RedundancyNode(parameter, &make_tuple_map);
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
      }
      if (root_->has_flag(NO_UPDATE) && IsPrimitiveCNode(make_tuple_user, prim::kPrimAddN)) {
        auto zeros = CreateZeroseOutput(input, 0);
        new_inputs.push_back(NewValueNode(zeros));
      }
    }
    auto new_make_tuple = fg->NewCNode(new_inputs);
    (void)manager_->Replace(make_tuple, new_make_tuple);
  }
}

void PipelinePostProcess::ModifyParameterList() {
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

void PipelineInterleave::CutBorder() {
  CreateSendReceiveGroup();
  MS_EXCEPTION_IF_NULL(shared_cell_);
  auto ret = shared_cell_->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::reverse(all_nodes.begin(), all_nodes.end());
  int64_t order = 0;
  for (auto &node : all_nodes) {
    auto stage_info = node->user_data<NodeStageInfo>();
    if (!node->isa<CNode>() || stage_info == nullptr || stage_info->stage() == -1 ||
        IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      continue;
    }
    // Modify for lizard cyclomatic complexity.
    CutBorderForNode(shared_cell_, node, &order);
  }
  RemoveMonadNode();
}

AnfNodePtr PipelinePostProcess::GetZeroOutputs(const FuncGraphPtr &graph) {
  auto real_kernel = GetRealKernelNode(graph->output(), -1);
  AnfNodePtr node = real_kernel.first;
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> out_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto each_out_shapes = GetNodeShape(cnode->input(i));
      if (each_out_shapes.size() > 1) {
        auto temp_tuple = CreateTupleZeroTensor(graph, cnode->input(i), each_out_shapes.size());
        (void)out_tuple_inputs.emplace_back(temp_tuple);
        continue;
      }
      (void)out_tuple_inputs.emplace_back(NewValueNode(CreateZeroseOutput(cnode->input(i), 0)));
    }
  }
  AnfNodePtr zero_outputs;
  if (out_tuple_inputs.size() > INDEX_ONE) {
    auto out_tuple = graph->NewCNode(out_tuple_inputs);
    return out_tuple;
  } else {
    auto out_shapes = GetNodeShape(node);
    AnfNodePtr out_tensor;
    if (out_shapes.size() > 1 && real_kernel.second == -1) {
      out_tensor = CreateTupleZeroTensor(graph, node, out_shapes.size());
    } else {
      out_tensor = NewValueNode(CreateZeroseOutput(node, 0));
    }
    return out_tensor;
  }
  return nullptr;
}

void PipelinePostProcess::SetNodeAbstract(const std::vector<AnfNodePtr> &nodes) {
  AbstractBasePtr abs;
  if (nodes.size() == 1) {
    auto cnode = nodes.front()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    abs = GetRealAbstract(cnode->input(INDEX_ONE));
  } else {
    AbstractBasePtrList abstract_list;
    abstract_list.resize(nodes.size());
    (void)std::transform(nodes.begin(), nodes.end(), abstract_list.begin(), [](const AnfNodePtr &node) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      return GetRealAbstract(cnode->input(INDEX_ONE));
    });
    abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  }
  for (auto &user : shared_cell_users_) {
    user->set_abstract(abs);
  }
}

void PipelinePostProcess::ModifySendRecvAttr(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto pre_node_pair = GetRealKernelNode(node, -1);
    auto pre_node = pre_node_pair.first;
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(node);
    Shape slice_shape;
    if (pre_node->isa<Parameter>()) {
      auto base_shape = pre_node->Shape();
      MS_EXCEPTION_IF_NULL(base_shape);
      auto shape_ptr = dyn_cast<abstract::Shape>(base_shape);
      MS_EXCEPTION_IF_NULL(shape_ptr);
      slice_shape = shape_ptr->shape();
      cnode->AddPrimalAttr(PIPELINE_PARAM, MakeValue(0));
      cnode->AddPrimalAttr(MICRO, MakeValue(int64_t(0)));
      cnode->set_user_data<AnfNode>(INPUT_PARAM, pre_node);
    } else {
      auto op_info = pre_node->cast<CNodePtr>()->user_data<OperatorInfo>();
      MS_EXCEPTION_IF_NULL(op_info);
      auto tensor_info = op_info->outputs_tensor_info();
      if (pre_node_pair.second != -1 && tensor_info.size() > 1) {
        slice_shape = tensor_info.at(pre_node_pair.second).slice_shape();
        node->set_user_data<TensorLayout>(
          std::make_shared<TensorLayout>(tensor_info.at(pre_node_pair.second).tensor_layout()));
      } else {
        slice_shape = tensor_info.at(0).slice_shape();
        node->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_info.at(0).tensor_layout()));
      }
    }
    auto abstract = node->abstract();
    abstract->set_shape(std::make_shared<abstract::Shape>(slice_shape));
    std::vector<ValuePtr> element;
    (void)std::transform(slice_shape.begin(), slice_shape.end(), std::back_inserter(element),
                         [](int elem) { return MakeValue(int64_t(elem)); });
    auto value = std::make_shared<ValueList>(element);
    prim->set_attr(SHAPE, value);
  }
}

static int64_t CalSrTag(int64_t order, int64_t micro, int64_t interleave_index) {
  return order * MAX_MICRO_BATCH_NUM * MAX_INTERLEAVE_NUM + interleave_index * MAX_INTERLEAVE_NUM + micro;
}

AnfNodePtr PipelinePostProcess::GenNewNodeFromOld(const AnfNodePtr &node, const AnfNodePtr &input, int64_t micro,
                                                  int64_t index) {
  const auto &old = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old);
  auto prim = GetCNodePrimitive(node);
  auto cloned_prim = prim->Clone();
  auto attrs = prim->attrs();
  auto order = GetValue<int64_t>(old->GetPrimalAttr(ORDER));
  auto sr_tag = CalSrTag(order, micro, index);
  attrs[SR_TAG] = MakeValue(sr_tag);
  cloned_prim->SetAttrs(attrs);
  std::vector<AnfNodePtr> new_node_input = {NewValueNode(cloned_prim), input};
  auto new_node = main_graph_->NewCNode(new_node_input);
  new_node->set_abstract(old->abstract());
  if (old->HasPrimalAttr(PIPELINE_PARAM)) {
    new_node->AddPrimalAttr(PIPELINE_PARAM, MakeValue(0));
  }
  new_node->set_primal_attrs(old->primal_attrs());
  new_node->AddPrimalAttr(ORDER, MakeValue(sr_tag));
  return new_node;
}

std::vector<AnfNodePtr> PipelinePostProcess::GenerateMainGraphSend(const std::vector<AnfNodePtr> &nodes,
                                                                   const AnfNodePtr &node, const ValuePtr &micro,
                                                                   const ValuePtr &index) {
  std::vector<AnfNodePtr> sends;
  auto index_value = GetValue<int64_t>(index);
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto send = nodes[i];
    auto csend = send->cast<CNodePtr>();
    if (csend->HasPrimalAttr(PIPELINE_PARAM)) {
      if (csend->HasPrimalAttr("send_once")) {
        continue;
      }
      auto param = csend->cast<CNodePtr>()->user_data<AnfNode>(INPUT_PARAM);
      csend->AddPrimalAttr("send_once", MakeValue(true));
      auto new_send = GenNewNodeFromOld(send, param, 0, 0);
      sends.emplace_back(new_send);
      continue;
    }
    auto micro_value = GetValue<int64_t>(micro);
    auto send_input = CreateTupleGetItemNode(main_graph_, node, i);
    auto new_send = GenNewNodeFromOld(send, send_input, micro_value, index_value)->cast<CNodePtr>();
    new_send->AddPrimalAttr(PIPELINE_END, micro);
    new_send->AddPrimalAttr(MICRO, micro);
    sends.emplace_back(new_send);
  }
  return sends;
}

AnfNodePtr PipelinePostProcess::GenerateMainGraphRecv(const AnfNodePtr &fg_node, const AnfNodePtr &recv) {
  auto cuser = fg_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cuser);
  auto crecv = recv->cast<CNodePtr>();
  AnfNodePtr new_recv;
  if (crecv->HasPrimalAttr(PIPELINE_PARAM)) {
    auto param = crecv->user_data<AnfNode>(INPUT_PARAM);
    MS_EXCEPTION_IF_NULL(param);
    new_recv = GenNewNodeFromOld(recv, param, 0, 0);
  } else {
    auto index = cuser->GetPrimalAttr(INDEX);
    MS_EXCEPTION_IF_NULL(index);
    auto index_value = GetValue<int64_t>(index);
    new_recv = GenNewNodeFromOld(recv, crecv->input(1), GetValue<int64_t>(cuser->GetPrimalAttr(MICRO)), index_value);
    new_recv->cast<CNodePtr>()->AddPrimalAttr(PIPELINE_BEGIN, cuser->GetPrimalAttr(MICRO));
  }
  new_recv->cast<CNodePtr>()->AddPrimalAttr(MICRO, cuser->GetPrimalAttr(MICRO));
  manager_->AddEdge(cuser, new_recv);
  return new_recv;
}

void PipelinePostProcess::Init(const std::vector<AnfNodePtr> &nodes) {
  shared_cell_ = nullptr;
  shared_cell_users_.clear();
  for (auto &node : nodes) {
    if ((IsPrimitiveCNode(node, prim::kPrimSend) || IsPrimitiveCNode(node, prim::kPrimReceive)) &&
        shared_cell_ == nullptr) {
      shared_cell_ = node->cast<CNodePtr>()->func_graph();
    }
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      auto cnode = node->cast<CNodePtr>();
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      main_graph_ = graph;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    chunk_num_ = (chunk + 1) > chunk_num_ ? (chunk + 1) : chunk_num_;
  }
  auto value_nodes = main_graph_->value_nodes();
  for (auto value_pair = value_nodes.cbegin(); value_pair != value_nodes.cend(); ++value_pair) {
    auto node = (*value_pair).first;
    if (!IsValueNode<FuncGraph>(node)) {
      continue;
    }
    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg != shared_cell_) {
      continue;
    }
    auto node_users = manager_->node_users()[node];
    for (auto &node_user : node_users) {
      auto user = node_user.first;
      if (user->func_graph() == main_graph_) {
        shared_cell_users_.emplace_back(user);
      }
    }
    break;
  }
}

void PipelinePostProcess::GetSendsRecvs(const FuncGraphPtr &fg, int64_t chunk, std::vector<AnfNodePtr> *recvs,
                                        std::vector<AnfNodePtr> *sends, std::vector<AnfNodePtr> *temp) {
  const auto &all_nodes = TopoSort(fg->get_return());
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!cnode->HasPrimalAttr(STAGE)) {
      continue;
    }
    auto stage_value = cnode->GetPrimalAttr(STAGE);
    if (stage_value && GetValue<int64_t>(stage_value) != stage_) {
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimSend) && GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK)) == chunk) {
      if (!cnode->HasPrimalAttr(PIPELINE_PARAM)) {
        temp->emplace_back(cnode->input(INDEX_ONE));
      }
      sends->emplace_back(node);
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimReceive) && GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK)) == chunk) {
      auto prim = GetCNodePrimitive(node);
      auto attrs = prim->attrs();
      auto zero_tensor = TensorConstructUtils::CreateZerosTensor(attrs[DTYPE]->cast<TypePtr>(), {1});
      manager_->SetEdge(node, 1, NewValueNode(zero_tensor));
      recvs->emplace_back(node);
    }
  }
  return;
}

void PipelinePostProcess::LabelInterleaveIndex() {
  std::vector<int64_t> micro_visited;
  for (auto &usr : shared_cell_users_) {
    int64_t index = 0;
    auto cusr = usr->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cusr);
    auto micro = cusr->GetPrimalAttr(MICRO);
    MS_EXCEPTION_IF_NULL(micro);
    auto micro_value = GetValue<int64_t>(micro);
    if (!std::count(micro_visited.begin(), micro_visited.end(), micro_value)) {
      micro_visited.emplace_back(micro_value);
    } else {
      index += 1;
    }
    cusr->AddPrimalAttr(INDEX, MakeValue(index));
  }
}

std::vector<AnfNodePtr> PipelinePostProcess::PartitionChunkGraph(const FuncGraphPtr &fg, int64_t chunk) {
  std::vector<AnfNodePtr> temp;
  std::vector<AnfNodePtr> recvs;
  std::vector<AnfNodePtr> sends;
  GetSendsRecvs(fg, chunk, &recvs, &sends, &temp);
  AnfNodePtr out;
  if (!temp.empty()) {
    out = CreateMakeTupleNode(fg, temp);
    manager_->Replace(fg->output(), out);
  }

  auto params = fg->parameters();
  std::vector<AnfNodePtr> new_params;
  auto node_users_map = manager_->node_users();
  std::vector<size_t> temp_index;
  for (size_t i = 0; i < params.size(); ++i) {
    auto param = params.at(i);
    if (node_users_map[param].size() == 0) {
      temp_index.emplace_back(i + 1);
      continue;
    }
    new_params.emplace_back(param);
  }
  for (auto &node : recvs) {
    auto crecv = node->cast<CNodePtr>();
    auto new_shared_cell_param = std::make_shared<Parameter>(fg);
    new_shared_cell_param->set_abstract(node->abstract());
    new_params.emplace_back(new_shared_cell_param);
    manager_->Replace(node, new_shared_cell_param);
  }
  manager_->SetParameters(fg, new_params);
  std::vector<AnfNodePtr> main_graph_sends;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> recv_map;
  for (auto &usr : shared_cell_users_) {
    auto cusr = usr->cast<CNodePtr>();
    std::vector<AnfNodePtr> usr_new_inputs = {NewValueNode(fg)};
    for (size_t i = 1; i < cusr->inputs().size(); ++i) {
      if (std::find(temp_index.begin(), temp_index.end(), i) == temp_index.end()) {
        usr_new_inputs.emplace_back(cusr->input(i));
      }
    }
    auto new_usr = main_graph_->NewCNode(usr_new_inputs);
    new_usr->set_primal_attrs(cusr->primal_attrs());
    new_usr->AddPrimalAttr(CHUNK, MakeValue(chunk));
    if (out != nullptr) {
      new_usr->set_abstract(out->abstract());
    }
    auto micro = cusr->GetPrimalAttr(MICRO);
    auto index = cusr->GetPrimalAttr(INDEX);
    auto temp_sends = GenerateMainGraphSend(sends, new_usr, micro, index);
    if (temp_sends.empty()) {
      if (stage_ != stage_num_ - 1) {
        MS_LOG(EXCEPTION) << "Some wrong with PipelineParallel.";
      }
      manager_->Replace(usr, new_usr);
    }
    main_graph_sends.insert(main_graph_sends.end(), temp_sends.begin(), temp_sends.end());
    for (auto &recv : recvs) {
      auto crecv = recv->cast<CNodePtr>();
      if (crecv->HasPrimalAttr(PIPELINE_PARAM)) {
        if (recv_map.find(recv) == recv_map.end()) {
          auto temp_recv = GenerateMainGraphRecv(new_usr, recv);
          recv_map[recv] = temp_recv;
          continue;
        }
        manager_->AddEdge(new_usr, recv_map[recv]);
        continue;
      }
      (void)GenerateMainGraphRecv(new_usr, recv);
    }
  }
  return main_graph_sends;
}

void PipelinePostProcess::GraphPartition(const std::vector<AnfNodePtr> &all_nodes) {
  LabelInterleaveIndex();
  std::vector<AnfNodePtr> send_ops;
  for (size_t i = 0; i < LongToSize(chunk_num_); ++i) {
    auto chunk_fg = shared_cell_;
    if (stage_ != stage_num_ - 1 || i != LongToSize(chunk_num_ - 1)) {
      chunk_fg = BasicClone(shared_cell_);
      chunk_fg->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);
      manager_->AddFuncGraph(chunk_fg);
    }
    auto sends = PartitionChunkGraph(chunk_fg, i);
    send_ops.insert(send_ops.begin(), sends.begin(), sends.end());
  }
  auto make_tuple = CreateMakeTupleNode(main_graph_, send_ops);
  auto outputs = GetZeroOutputs(main_graph_);
  if (stage_ == stage_num_ - 1) {
    outputs = main_graph_->output();
  }
  std::vector<AnfNodePtr> out = {NewValueNode(prim::kPrimDepend), outputs, make_tuple};
  auto out_node = main_graph_->NewCNode(out);
  (void)manager_->Replace(main_graph_->output(), out_node);
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
    if (!cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      continue;
    }
    auto param = cnode->input(1);
    if (IsPrimitiveCNode(param, prim::kPrimVirtualAssignAdd)) {
      param = param->cast<CNodePtr>()->input(1);
    }
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
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
      auto slice_shape = shape_ptr->shape();
      auto prim = GetCNodePrimitive(cnode);
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

bool PipelineInterleave::HasNoUpdateParameter() {
  auto parameters = root_->parameters();
  for (auto &parameter : parameters) {
    if (ParameterIsCloned(parameter)) {
      continue;
    }
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

void PipelineInterleave::FreezeGradient() {
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
        std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), out, temp};
        auto new_node = root_->NewCNode(depend_input);
        manager_->Replace(out, new_node);
        break;
      }
      break;
    }
    for (auto &node : nodes) {
      if (!IsPrimitiveCNode(node, prim::kPrimNPUGetFloatStatusV2)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto out_cnode = root_->output()->cast<CNodePtr>();
      auto grads = out_cnode->input(INDEX_TWO);
      std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), cnode->input(1), grads};
      auto new_node = root_->NewCNode(depend_input);
      new_node->set_abstract(cnode->input(1)->abstract());
      manager_->Replace(cnode->input(1), new_node);
      break;
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

static bool NeedAttach(const FuncGraphManagerPtr &manager) {
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

bool IsolatedNodeAttach(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  if (root->has_flag(HAS_ATTACHED)) {
    return false;
  }
  root->set_flag(HAS_ATTACHED, true);
  auto manager = root->manager();
  if (!NeedAttach(manager)) {
    return false;
  }
  auto ret_after = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = DeepScopedGraphSearch(ret_after);
  const auto &node_users_map = manager->node_users();
  std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple)};
  FuncGraphPtr main_graph;
  FuncGraphPtr grad_graph;
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<FuncGraph>(cnode->input(0))) {
      continue;
    }
    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto sub_graph_output = graph->output();
    if (!IsPrimitiveCNode(sub_graph_output, prim::kPrimMakeTuple)) {
      continue;
    }
    auto csub_graph_output = sub_graph_output->cast<CNodePtr>();
    if (!IsPrimitiveCNode(csub_graph_output->input(1), prim::kPrimReceive)) {
      continue;
    }
    auto call_node_input = cnode->input(1);
    if (!IsValueNode<tensor::Tensor>(call_node_input)) {
      continue;
    }
    auto call_node_users = node_users_map.at(node);
    if (call_node_users.size() != 1) {
      continue;
    }
    auto usr_node = call_node_users.begin()->first;
    if (!IsPrimitiveCNode(usr_node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto get_item_usrs = node_users_map.at(usr_node);
    std::vector<AnfNodePtr> addn_input = {NewValueNode(prim::kPrimAddN)};
    main_graph = node->func_graph();
    for (auto &get_item_usr_pair : get_item_usrs) {
      auto get_item_usr = get_item_usr_pair.first;
      auto grad_node = GetDout(get_item_usr, node_users_map);
      if (grad_graph == nullptr) {
        grad_graph = grad_node->func_graph();
      } else {
        if (grad_graph != grad_node->func_graph()) {
          MS_LOG(EXCEPTION) << "Got Wrong Grad graph when attached Receive's grad, Maybe don't use lazy inline.";
        }
      }
      std::vector<AnfNodePtr> new_get_item_input = {NewValueNode(prim::kPrimTupleGetItem), grad_node,
                                                    NewValueNode(MakeValue(SizeToLong(get_item_usr_pair.second)))};
      auto new_get_item = grad_graph->NewCNode(new_get_item_input);
      addn_input.emplace_back(new_get_item);
    }
    AnfNodePtr temp;
    if (addn_input.size() > SIZE_TWO) {
      temp = grad_graph->NewCNode(addn_input);
    } else {
      temp = addn_input.at(1);
    }
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
