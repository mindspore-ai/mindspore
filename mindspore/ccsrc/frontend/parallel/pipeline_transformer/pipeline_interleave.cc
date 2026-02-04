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
constexpr char kSharedParamMirrorNode[] = "shared_param_mirror_node";

static AbstractBasePtr GetRealAbstract(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    MS_EXCEPTION_IF_NULL(node->cast<CNodePtr>());
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
  if (rank_list.size() == g_device_manager->DeviceNum()) {
    group_ = g_device_manager->world_group();
    return;
  }
  auto dev_list = g_device_manager->CreateDeviceListByRankList(rank_list);
  Group forward_send_group;
  auto forward_send_group_name = g_device_manager->GenerateGroupNameByRanks(rank_list) + SEND;
  if (g_device_manager->CreateGroup(forward_send_group_name, dev_list, &forward_send_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create forward Send communication group failed, the rank list is: " << rank_list;
  }
  group_ = forward_send_group_name;
}

ValuePtr PipelineInterleave::SetMicroBatch(const AnfNodePtr &node, int64_t micro_size, size_t batch_axis) const {
  if (!IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Can't find MicroBatch information.";
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  int64_t micro = 0;
  auto value = GetValueNode(cnode->input(2));
  if (value != nullptr) {
    auto tuple = GetValue<std::vector<int64_t>>(value);  // begin
    auto input_tmp = GetNodeShape(cnode->input(1));
    auto input_shape = input_tmp.at(0);
    auto slice_batch_size = input_shape.at(batch_axis);  // betch shape
    if (slice_batch_size == 0) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "slice_batch_size should be a positive integer, but got "
                                         << slice_batch_size;
    }
    micro = tuple.at(batch_axis) * micro_size / slice_batch_size;  // micro-index
  } else {
    // dynamic shape
    // if micro is not 1: stridedslice --> maketuple --> scalarmul --> micro
    // if micro is 1: stridedslice --> maketuple --> scalarfloordiv
    if (!IsPrimitiveCNode(cnode->input(2), prim::kPrimMakeTuple)) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The begin of stridedslice is not constant value, and not make tuple";
    }
    auto make_tuple_cnode = cnode->input(2)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_cnode);
    MS_EXCEPTION_IF_NULL(make_tuple_cnode->input(1));
    if (IsPrimitiveCNode(make_tuple_cnode->input(1), prim::kPrimScalarMul)) {
      auto scalar_mul_cnode = make_tuple_cnode->input(1)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(scalar_mul_cnode);
      auto mul_value = GetValueNode(scalar_mul_cnode->input(2));
      micro = GetValue<int64_t>(mul_value);
    } else if (IsPrimitiveCNode(make_tuple_cnode->input(1), prim::kPrimScalarFloorDiv)) {
      micro = 1;
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, make_tuple_cnode) << "Can not find the micro info, the input op of make tuple is "
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
  auto scheduler = parallel::ParallelContext::GetInstance()->pipeline_scheduler();
  if (scheduler == ZBV) {
    is_v_shape_ = true;
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

// find StridedSlice by DFS
void PipelineInterleave::FindStridedSliceNodes(const AnfNodePtr &node, AnfNodeSet *strided_slice_nodes) const {
  if (IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
    strided_slice_nodes->push_back(node);
    return;
  }
  if (!IsPrimitiveCNode(node, prim::kPrimDepend) && !IsPrimitiveCNode(node, prim::kPrimInsertGradientOf) &&
      !IsPrimitiveCNode(node, prim::kPrimDumpGradient)) {
    return;
  }
  auto node_user = manager_->node_users()[node];
  for (const auto &user : node_user) {
    FindStridedSliceNodes(user.first, strided_slice_nodes);
  }
}

size_t PipelineInterleave::GetBatchAxisForInput(const AnfNodeIndexSet &input_node_users) const {
  Shapes inputs_tuple;
  for (const auto &input_node_user : input_node_users) {
    AnfNodeSet strided_slice_nodes;
    FindStridedSliceNodes(input_node_user.first, &strided_slice_nodes);
    if (strided_slice_nodes.size() == 0) {
      return 0;  // simply return 0 when dynamic shape
    }
    for (const auto &node : strided_slice_nodes) {
      auto cnode = node->cast<CNodePtr>();
      auto value = GetValueNode(cnode->input(2));
      if (value == nullptr) {
        return 0;  // simply return 0 when dynamic shape
      }
      auto tuple = GetValue<std::vector<int64_t>>(value);
      inputs_tuple.push_back(tuple);
    }
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
  if (batch_axis_count == kSizeZero) {
    MS_LOG(EXCEPTION) << "For pipeline parallelism, batch data must be split into micro batch along one dimension. "
                      << "Please check the implementation of the data partitioning micro batch in the script. or "
                         "setting micro_batch_num > 1.";
  } else if (batch_axis_count > kSizeOne) {
    MS_LOG(EXCEPTION)
      << "For pipeline parallelism, batch data must be split into micro batch along one dimension. "
         "However, it is detected that you have divided micro batch along "
      << batch_axis_count
      << " dimensions, please check the implementation of the data partitioning micro batch in the script.";
  }
  return batch_axis;
}

size_t PipelineInterleave::MicroSize(const AnfNodeIndexSet &input_node_users) const {
  size_t micro_size = 0;
  for (const auto &input_node_user : input_node_users) {
    auto node = input_node_user.first;
    if (IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
      micro_size++;
    } else if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimInsertGradientOf) ||
               IsPrimitiveCNode(node, prim::kPrimDumpGradient)) {
      auto next_node_user = manager_->node_users()[node];
      micro_size += MicroSize(next_node_user);
    }
  }

  return micro_size;
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
      for (const auto &data_user : data_users) {
        auto data_node = data_user.first;
        if (IsPrimitiveCNode(data_node, prim::kPrimTensorDump)) {
          continue;
        }
        node_first = data_user.first;
        break;
      }
      if (!IsPrimitiveCNode(node_first, prim::kPrimStridedSlice) && !IsPrimitiveCNode(node_first, prim::kPrimShape)) {
        data_users.clear();
        data_users = node_user_map[node_first];
      }
      auto micro_size = int64_t(MicroSize(data_users));
      auto stage_num = g_device_manager->stage_num();
      auto scheduler = parallel::ParallelContext::GetInstance()->pipeline_scheduler();
      int64_t kSizeTwo = 2;  // 2: for double
      if ((scheduler == ZBV) && (micro_size < stage_num * kSizeTwo)) {
        MS_LOG(EXCEPTION)
          << "For zero_bubble_v scheduler, micro_size must be greater than or equal to twice stage_num. Got micro_size:"
          << micro_size << ", stage_num:" << stage_num;
      }
      micro_size_ = micro_size;
      auto batch_axis = GetBatchAxisForInput(data_users);
      MS_LOG(INFO) << "For the "
                   << GetSerialNumberString(
                        GetValue<int64_t>(GetValueNode(node_user.first->cast<CNodePtr>()->input(kIndex2))))
                   << "input, batch axis is " << batch_axis << ", micro size is : " << micro_size;
      for (auto &data_user : data_users) {
        AnfNodeSet strided_slice_nodes;
        FindStridedSliceNodes(data_user.first, &strided_slice_nodes);
        if (strided_slice_nodes.size() == 0) {
          continue;
        }
        for (const auto &strided_slice_node : strided_slice_nodes) {
          auto micro = SetMicroBatch(strided_slice_node, micro_size, batch_axis);
          SetStridedSliceStrategy(strided_slice_node);
          auto cnode = strided_slice_node->cast<CNodePtr>();
          BroadCastMicroBatch(cnode, &node_user_map, micro, 0);
        }
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

FuncGraphPtr GetShardedFuncGraph(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const size_t kShardInputSize = 6;
  if (cnode->size() < kShardInputSize) {
    return nullptr;
  }
  AnfNodePtr input0 = cnode->input(0);
  if (!input0->isa<ValueNode>()) {
    return nullptr;
  }
  auto val_node0 = input0->cast<ValueNodePtr>();
  auto prim_val0 = GetValueNode<PrimitivePtr>(val_node0);
  if (prim_val0 != nullptr && prim_val0->name() != prim::kPrimShard->name()) {
    return nullptr;
  }
  AnfNodePtr input1 = cnode->input(1);
  if (!input1->isa<ValueNode>()) {
    return nullptr;
  }
  auto val_node1 = input1->cast<ValueNodePtr>();
  FuncGraphPtr wrapped_fg = GetValueNode<FuncGraphPtr>(val_node1);
  return wrapped_fg;
}

void ContainsStageInSubGraph(const FuncGraphPtr &sub_graph, const std::vector<FuncGraphPtr> &shard_subgraph_list,
                             const std::string &error_msg) {
  if (sub_graph->stage() != -1) {
    MS_LOG(EXCEPTION) << error_msg << "a sub cell '" << sub_graph->ToString() << "' with pipeline stage "
                      << sub_graph->stage() << " is not supported yet. Please Check!";
  }
  auto value_nodes = sub_graph->value_nodes();
  for (const auto &value_pair : value_nodes) {
    auto node = value_pair.first;
    if (!IsValueNode<FuncGraph>(node)) {
      continue;
    }
    auto nested_fg = GetValueNode<FuncGraphPtr>(node);
    if (nested_fg == nullptr) {
      continue;
    }
    if (std::find(shard_subgraph_list.begin(), shard_subgraph_list.end(), nested_fg) != shard_subgraph_list.end()) {
      continue;
    }
    MS_LOG(INFO) << "[sub_graph] " << sub_graph->ToString() << ", [nested_fg] " << nested_fg->ToString();
    ContainsStageInSubGraph(nested_fg, shard_subgraph_list, error_msg);
  }
}

void TraverseAndCheckShard(const FuncGraphManagerPtr &manager) {
  std::vector<FuncGraphPtr> shard_subgraph_list;
  auto all_graphs = manager->func_graphs();
  for (const auto &fg : all_graphs) {
    for (const auto &node : fg->nodes()) {
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto wrapped_fg = GetShardedFuncGraph(cnode);
      if (wrapped_fg == nullptr) {
        continue;
      }
      if (wrapped_fg->stage() != -1) {
        MS_LOG(EXCEPTION) << "For sharded cell '" << wrapped_fg->ToString() << ", "
                          << "pipeline stage " << wrapped_fg->stage() << " is not supported yet. Please Check!";
      }
      if (std::find(shard_subgraph_list.begin(), shard_subgraph_list.end(), wrapped_fg) == shard_subgraph_list.end()) {
        shard_subgraph_list.push_back(wrapped_fg);
      }
    }
  }

  for (const auto &shard_subgraph : shard_subgraph_list) {
    const std::string error_msg = "For sharded cell '" + shard_subgraph->ToString() + "', ";
    ContainsStageInSubGraph(shard_subgraph, shard_subgraph_list, error_msg);
  }
}

void PipelineInterleave::Coloring() {
  TraverseAndCheckShard(manager_);
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
        MS_EXCEPTION_IF_NULL(graph);
        if (graph->stage() == -1) {
          continue;
        }
        (void)stage_set.insert(graph->stage());
        auto node_users = manager_->node_users()[node];
        for (auto &user_pair : node_users) {
          auto user_node = user_pair.first->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(user_node);
          auto stage_info = std::make_shared<NodeStageInfo>(graph->stage());
          if (graph->segment() != -1 && is_v_shape_) {
            stage_info->set_chunk(graph->segment());
          }
          user_node->set_user_data<NodeStageInfo>(stage_info);
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

void PipelineInterleave::BroadCastGraphStage(const FuncGraphPtr &fg) {
  if (fg == root_ || fg == main_graph_ || fg == shared_cell_) {
    return;
  }
  auto stage = fg->stage();
  auto value_nodes = fg->value_nodes();
  for (const auto &value_pair : value_nodes) {
    auto node = value_pair.first;
    if (IsValueNode<FuncGraph>(node)) {
      auto sub_graph = GetValueNode<FuncGraphPtr>(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      sub_graph->set_stage(stage);
      BroadCastGraphStage(sub_graph);
    }
  }
}

static bool IsNodeSkippable(const AnfNodePtr &node) {
  return !node->isa<CNode>() || node->user_data<NodeStageInfo>() == nullptr ||
         node->user_data<NodeStageInfo>()->stage() == -1 || IsPrimitiveCNode(node, prim::kPrimUpdateState);
}

struct StageChunkUpdate {
  bool need_coloring;
  int64_t new_chunk;
};

static void ZBVErrorCheck(int64_t stage, int64_t chunk, int64_t user_stage, int64_t user_chunk) {
  // case1: stage > user_stage, chunk >= user_chunk
  constexpr int64_t MAX_CHUNK_NUM = 1;
  if (chunk > MAX_CHUNK_NUM || user_chunk > MAX_CHUNK_NUM) {
    MS_LOG(EXCEPTION) << "Segment only support 0 and 1 in Zero Bubble V scheduler. "
                         "Got layer 's segment:"
                      << chunk << ". next layer' s segment : " << user_chunk;
  }
  if (chunk > user_chunk) {
    MS_LOG(EXCEPTION) << "Segment must be configured in ascending order. Got layer's segment:" << chunk
                      << ". next layer's segment:" << user_chunk;
  }
  if ((stage > user_stage) && (user_chunk != 1)) {
    MS_LOG(EXCEPTION) << "The stage and segment configuration is incorrect. When the segment is 0, the stage "
                         "must be configured in ascending order.When the segment is 1, "
                         "the stage must be configured in descending order.Got layer's segment:"
                      << chunk << ", stage_id:" << stage << ". next layer's segment:" << user_chunk
                      << ", stage_id:" << user_stage;
  }
  if ((stage < user_stage) && (chunk != 0)) {
    MS_LOG(EXCEPTION) << "The stage and segment configuration is incorrect. When the segment is 0, the stage "
                         "must be configured in ascending order.When the segment is 1, "
                         "the stage must be configured in descending order.Got layer's segment:"
                      << chunk << ", stage_id:" << stage << ". next layer's segment:" << user_chunk
                      << ", stage_id:" << user_stage;
  }
}

static StageChunkUpdate UpdateUserStageChunk(const std::shared_ptr<NodeStageInfo> &stage_info, const CNodePtr &cnode,
                                             const CNodePtr &user_node, int64_t stage, int64_t chunk, bool is_v_shape) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(user_node);
  bool need_coloring = false;
  auto user_stage_info = user_node->user_data<NodeStageInfo>();
  if (user_stage_info == nullptr) {
    user_node->set_user_data<NodeStageInfo>(std::make_shared<NodeStageInfo>(stage, chunk));
    need_coloring = true;
    user_node->AddPrimalAttr(CHUNK, MakeValue(chunk));
    user_node->AddPrimalAttr(STAGE, MakeValue(stage));
    return {need_coloring, chunk};
  }

  auto user_node_stage = user_stage_info->stage();
  auto user_node_chunk = user_stage_info->chunk();
  if (is_v_shape) {
    ZBVErrorCheck(stage, chunk, user_node_stage, user_node_chunk);
    return {need_coloring, chunk};
  }
  if (stage == user_node_stage) {
    if (chunk > user_node_chunk) {
      user_stage_info->set_chunk(chunk);
      need_coloring = true;
      user_node->AddPrimalAttr(CHUNK, MakeValue(chunk));
      user_node->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
      return {need_coloring, chunk};
    }
    if (chunk < user_node_chunk) {
      stage_info->set_chunk(user_node_chunk);
      chunk = user_node_chunk;
      need_coloring = true;
      cnode->AddPrimalAttr(CHUNK, MakeValue(chunk));
      cnode->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
      return {need_coloring, chunk};
    }
  }

  if (stage > user_node_stage) {
    const auto pc = parallel::ParallelContext::GetInstance();
    if (!pc->pipeline_interleave_temp() && IsValueNode<FuncGraph>(user_node->input(0))) {
      MS_LOG_WITH_NODE(EXCEPTION, user_node) << "The stage setting is incorrect. PreNode's stage:" << stage
                                             << " is larger than NextNode's stage:" << user_node_stage;
    }
    if ((chunk >= user_node_chunk)) {
      user_stage_info->set_chunk(chunk + 1);
      need_coloring = true;
      user_node->AddPrimalAttr(CHUNK, MakeValue(chunk + 1));
      user_node->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
      return {need_coloring, chunk};
    }
  }

  if ((stage < user_node_stage) && (chunk > user_node_chunk)) {
    user_stage_info->set_chunk(chunk);
    need_coloring = true;
    user_node->AddPrimalAttr(CHUNK, MakeValue(chunk));
    user_node->AddPrimalAttr(STAGE, MakeValue(user_node_stage));
    return {need_coloring, chunk};
  }

  return {need_coloring, chunk};
}

void PipelineInterleave::BroadCastColoring() {
  auto need_coloring = true;
  auto all_nodes = shared_cell_->nodes();
  auto node_users = manager_->node_users();

  while (need_coloring) {
    need_coloring = false;
    for (auto node = all_nodes.cbegin(); node != all_nodes.cend(); ++node) {
      if (IsNodeSkippable(*node)) {
        continue;
      }

      auto stage_info = (*node)->user_data<NodeStageInfo>();
      auto cnode = (*node)->cast<CNodePtr>();
      auto stage = stage_info->stage();
      auto chunk = stage_info->chunk();
      for (auto &user_pair : node_users[*node]) {
        auto user_node = user_pair.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(user_node);
        const auto res = UpdateUserStageChunk(stage_info, cnode, user_node, stage, chunk, is_v_shape_);
        if (res.need_coloring) {
          need_coloring = true;
        }
        chunk = res.new_chunk;
      }
    }
  }

  for (const auto &fg : manager_->func_graphs()) {
    if (fg->stage() >= 0) {
      BroadCastGraphStage(fg);
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
  for (const auto &load : loads) {
    const auto &iter = node_users_map.find(load);
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
  }
  param_stage_ = *parameter_stage->begin();
  return true;
}

void PipelineInterleave::InsertSendReceiveForParameter(const AnfNodePtr &param, const AnfNodePtr &node,
                                                       int64_t src_stage, int64_t dst_stage, int64_t chunk,
                                                       int64_t index, int64_t order) {
  if (src_stage != stage_ && dst_stage != stage_) {
    return;
  }
  Attr attr_tag = std::make_pair(SR_TAG, MakeValue(0));
  Attr attr_rank = std::make_pair(DEST_RANK, MakeValue(dst_stage));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_));
  Attr attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_));
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_group, attr_group_back};
  auto send_op = CreateOpInstance(attrs, SEND, SEND);
  auto send_node = NewValueNode(send_op);
  std::vector<AnfNodePtr> send_input = {send_node, param};
  auto graph = shared_cell_;
  auto send = graph->NewCNode(send_input);
  send->set_abstract(param->abstract());
  send->AddPrimalAttr(CHUNK, MakeValue(chunk));
  send->AddPrimalAttr(STAGE, MakeValue(src_stage));
  send->AddPrimalAttr(ORDER, MakeValue(order));
  auto is_freeze = IsFreezedGradGraph(param);
  send->AddPrimalAttr(FREEZE, MakeValue(is_freeze));
  attr_rank = std::make_pair(SRC_RANK, MakeValue(src_stage));
  auto shape_type_pair = GetShapeType(param, {1}, 0);
  Attr attr_shape = std::make_pair(SHAPE, shape_type_pair.first);
  Attr attr_dtype = std::make_pair(DTYPE, shape_type_pair.second);
  auto send_prim = GetCNodePrimitive(send);
  MS_EXCEPTION_IF_NULL(send_prim);
  auto rank_list = g_device_manager->GetDeviceListBetweenStage();
  send_prim->set_attr(DST_GLOBAL_RANK, MakeValue(rank_list[dst_stage]));
  send_prim->set_attr(DTYPE, shape_type_pair.second);
  OperatorAttrs attrs_recv = {attr_tag, attr_rank, attr_shape, attr_dtype, attr_group, attr_group_back};
  auto recv_op = CreateOpInstance(attrs_recv, RECEIVE, RECEIVE);
  std::vector<AnfNodePtr> recv_input = {NewValueNode(recv_op), send};
  auto recv = graph->NewCNode(recv_input);
  auto recv_prim = GetCNodePrimitive(recv);
  MS_EXCEPTION_IF_NULL(recv_prim);
  recv_prim->set_attr(SRC_GLOBAL_RANK, MakeValue(rank_list[src_stage]));
  recv->set_abstract(param->abstract());
  recv->AddPrimalAttr(CHUNK, MakeValue(chunk));
  recv->AddPrimalAttr(STAGE, MakeValue(dst_stage));
  recv->AddPrimalAttr(ORDER, MakeValue(order));
  recv->AddPrimalAttr(FREEZE, MakeValue(is_freeze));
  manager_->SetEdge(node, index, recv);
}

std::shared_ptr<NodeStageInfo> PipelineInterleave::GetStageInfoByGraph(const FuncGraphPtr &fg) {
  auto fg_users = fg->func_graph_cnodes_index();
  for (const auto &fg_user : fg_users) {
    if (fg_user.first->second != 0) {
      continue;
    }
    auto call_node = fg_user.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(call_node);
    auto stage_info = call_node->user_data<NodeStageInfo>();
    if (stage_info != nullptr) {
      return stage_info;
    }
  }
  return nullptr;
}

void PipelineInterleave::InsertSendReceiveForSharedParam(const AnfNodePtr &parameter, const AnfNodePtr &argument,
                                                         int64_t *order) {
  auto stage_set = parameter_color_map_.at(parameter);
  if (stage_set.size() <= 1) {
    return;
  }
  auto src_stage = *stage_set.begin();
  auto loads = GetLoadNodeByParam(argument);
  auto node_users_map = manager_->node_users();
  for (const auto &load : loads) {
    auto param_users = node_users_map.at(load);
    for (const auto &param_user : param_users) {
      auto cuser = param_user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cuser);
      auto stage_info = cuser->user_data<NodeStageInfo>();
      if (cuser->func_graph() == shared_cell_ && stage_info == nullptr) {
        continue;
      }
      if (stage_info != nullptr) {
        auto stage = stage_info->stage();
        if (stage == src_stage) {
          continue;
        }
        auto chunk = stage_info->chunk();
        InsertSendReceiveForParameter(parameter, cuser, src_stage, stage, chunk, param_user.second, *order);
        (*order) += 1;
        continue;
      }
      if (cuser->func_graph() != shared_cell_) {
        auto stage = cuser->func_graph()->stage();
        int64_t chunk = 0;
        stage_info = GetStageInfoByGraph(cuser->func_graph());
        if (stage_info != nullptr) {
          stage = stage_info->stage();
          chunk = stage_info->chunk();
        }
        if (stage != src_stage) {
          InsertSendReceiveForParameter(parameter, cuser, src_stage, stage, chunk, param_user.second, *order);
          (*order) += 1;
        }
      }
    }
  }
}

void PipelineInterleave::HandleSharedParam(int64_t *order) {
  auto parameters = shared_cell_->parameters();
  auto fg_users = shared_cell_->func_graph_cnodes_index();
  CNodePtr call_node;
  auto node_users_map = manager_->node_users();
  for (const auto &fg_user : fg_users) {
    if (fg_user.first->second != 0) {
      continue;
    }
    call_node = fg_user.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(call_node);
    if (call_node->func_graph() != main_graph_) {
      call_node = nullptr;
      continue;
    }
    break;
  }
  MS_EXCEPTION_IF_NULL(call_node);
  for (size_t i = 1; i < call_node->inputs().size(); ++i) {
    auto real_node = GetRealKernelNode(call_node->input(i), -1, nullptr, false).first;
    if (real_node == nullptr) {
      continue;
    }
    if (!real_node->isa<Parameter>() || real_node->func_graph() != root_) {
      continue;
    }
    auto param = parameters[i - 1];
    InsertSendReceiveForSharedParam(real_node, param, order);
  }
}

mindspore::HashMap<int64_t, std::vector<int64_t>> PipelineInterleave::BuildStageRanksMap() const {
  mindspore::HashMap<int64_t, std::vector<int64_t>> stage_ranks_map;
  const auto &pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  for (int64_t stage_id = 0; stage_id < pipeline_stages; ++stage_id) {
    std::vector<int64_t> rank_list;
    for (int64_t i = 0; i < per_stage_rank_num_; ++i) {
      rank_list.push_back(stage_id * per_stage_rank_num_ + i);
    }
    stage_ranks_map[stage_id] = rank_list;
  }
  return stage_ranks_map;
}

void PipelineInterleave::ParameterColoring() {
  auto parameters = root_->parameters();
  auto &node_users_map = manager_->node_users();
  const auto &share_cell_parameters = shared_cell_->parameters();
  const auto &stage_ranks_map = BuildStageRanksMap();
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
    const auto &param = parameter->cast<ParameterPtr>();
    if (param == nullptr) {
      continue;
    }
    const auto &param_name = param->name();
    auto starts_with = [](const std::string &s, const char *p) { return s.rfind(p, 0) == 0; };
    bool isParamGradsOpts = starts_with(param_name, "accu_grads.") || starts_with(param_name, "moments.") ||
                            starts_with(param_name, "adam_m.") || starts_with(param_name, "adam_v.");
    if (!isParamGradsOpts && param_stage_ != -1) {
      const int64_t stage_id = param_stage_;
      const auto &cur_stage_ranks = stage_ranks_map.at(stage_id);
      StrategyLayout::GetInstance()->SetParamStageIdRanks(param_name, stage_id, cur_stage_ranks);
    }
  }
}

void PipelinePostProcess::RemoveMonadNodeBetweenStage(const CNodePtr &cnode) {
  auto node_users = manager_->node_users()[cnode];
  for (const auto &user_node_pair : node_users) {
    auto user_cnode = user_node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    for (const auto &input : user_cnode->inputs()) {
      if (IsPrimitiveCNode(input, prim::kPrimReceive)) {
        auto monad_node = NewValueNode(kUMonad);
        monad_node->set_abstract(kUMonad->ToAbstract());
        auto abs = cnode->abstract();
        MS_EXCEPTION_IF_NULL(abs);
        if (abs->isa<abstract::AbstractIOMonad>()) {
          monad_node = NewValueNode(kIOMonad);
          monad_node->set_abstract(kIOMonad->ToAbstract());
        }
        (void)manager_->SetEdge(user_node_pair.first, user_node_pair.second, monad_node);
        break;
      }
    }
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
  const auto &vec = out_shapes.at(index);
  const auto &it = std::find(vec.begin(), vec.end(), -1);
  if (it != vec.end()) {
    MS_LOG(EXCEPTION) << "Under pipeline parallelism, the output of the network does not support dynamic shapes.";
  }
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

void PipelineInterleave::InsertSendReceive(const AnfNodePtr &node, const AnfNodePtr &user_node, int64_t order,
                                           int64_t index, bool is_v_shape) {
  auto node_stage_info = node->user_data<NodeStageInfo>();
  auto user_node_stage_info = user_node->user_data<NodeStageInfo>();
  MS_EXCEPTION_IF_NULL(node_stage_info);
  auto node_stage = node_stage_info->stage();
  auto user_stage = user_node_stage_info->stage();
  Attr attr_tag = std::make_pair(SR_TAG, MakeValue(0));
  Attr attr_rank = std::make_pair(DEST_RANK, MakeValue(user_stage));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_));
  Attr attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_));
  if (node_stage > user_stage) {
    attr_group = std::make_pair(GROUP, MakeValue(group_));
    attr_group_back = std::make_pair(GROUP_BACK, MakeValue(group_));
  }
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_group, attr_group_back};
  auto send_op = CreateOpInstance(attrs, SEND, SEND);
  auto send_node = NewValueNode(send_op);
  std::vector<AnfNodePtr> send_input = {send_node, node};
  auto graph = shared_cell_;
  auto send = graph->NewCNode(send_input);
  send->set_user_data<NodeStageInfo>(node_stage_info);
  send->set_abstract(node->abstract());
  if (!is_vpp_ && node_stage_info->chunk() >= 1) {
    is_vpp_ = true;
  }
  send->AddPrimalAttr(CHUNK, MakeValue(node_stage_info->chunk()));
  send->AddPrimalAttr(STAGE, MakeValue(node_stage_info->stage()));
  send->AddPrimalAttr(ORDER, MakeValue(order));
  send->AddPrimalAttr(V_SHAPE, MakeValue(is_v_shape));

  auto is_freeze = IsFreezedGradGraph(node) || IsFreezedGradGraph(user_node);
  send->AddPrimalAttr(FREEZE, MakeValue(is_freeze));

  attr_rank = std::make_pair(SRC_RANK, MakeValue(node_stage));
  auto shape_type_pair = GetShapeType(node, {1}, 0);
  Attr attr_shape = std::make_pair(SHAPE, shape_type_pair.first);
  Attr attr_dtype = std::make_pair(DTYPE, shape_type_pair.second);
  auto send_prim = GetCNodePrimitive(send);
  MS_EXCEPTION_IF_NULL(send_prim);
  auto rank_list = g_device_manager->GetDeviceListBetweenStage();
  send_prim->set_attr(DST_GLOBAL_RANK, MakeValue(rank_list[user_stage]));
  send_prim->set_attr(DTYPE, shape_type_pair.second);
  OperatorAttrs attrs_recv = {attr_tag, attr_rank, attr_shape, attr_dtype, attr_group, attr_group_back};
  auto recv_op = CreateOpInstance(attrs_recv, RECEIVE, RECEIVE);
  std::vector<AnfNodePtr> recv_input = {NewValueNode(recv_op), send};
  auto recv = graph->NewCNode(recv_input);
  auto recv_prim = GetCNodePrimitive(recv);
  MS_EXCEPTION_IF_NULL(recv_prim);
  recv_prim->set_attr(SRC_GLOBAL_RANK, MakeValue(rank_list[node_stage]));
  recv->set_abstract(node->abstract());
  recv->set_user_data<NodeStageInfo>(user_node_stage_info);
  if (!is_vpp_ && node_stage_info->chunk() >= 1) {
    is_vpp_ = true;
  }
  recv->AddPrimalAttr(CHUNK, MakeValue(user_node_stage_info->chunk()));
  recv->AddPrimalAttr(STAGE, MakeValue(user_node_stage_info->stage()));
  recv->AddPrimalAttr(ORDER, MakeValue(order));
  recv->AddPrimalAttr(V_SHAPE, MakeValue(is_v_shape));
  recv->AddPrimalAttr(FREEZE, MakeValue(is_freeze));
  auto micro = user_node->cast<CNodePtr>()->GetPrimalAttr(MICRO);
  if (micro != nullptr) {
    recv->AddPrimalAttr(MICRO, micro);
  }
  manager_->SetEdge(user_node, index, recv);
}

void PipelineInterleave::CutBorderForNode(const FuncGraphPtr &graph, const AnfNodePtr &node, int64_t *order) {
  auto stage_info = node->user_data<NodeStageInfo>();
  MS_EXCEPTION_IF_NULL(stage_info);
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
    MS_EXCEPTION_IF_NULL(user_node->cast<CNodePtr>());
    auto micro = user_node->cast<CNodePtr>()->GetPrimalAttr(MICRO);
    if (!micro) {
      MS_LOG(INFO) << "Can't find micro_batch information, use micro(0)";
      micro = MakeValue(int64_t(0));
    }
    auto stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
    auto node_chunk = stage_info->chunk();
    auto user_node_chunk = user_stage_info->chunk();
    if (is_v_shape_ && node_chunk < user_node_chunk) {
      if (stage_ == stage_num - 1) {
        InsertSendReceive(node, user_node, *order, user_pair.second, true);
      }
      (*order) += 1;
      continue;
    }
    if (node_stage != user_node_stage) {
      InsertSendReceive(node, user_node, *order, user_pair.second);
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
    MS_EXCEPTION_IF_NULL(cnode);
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

void PipelineInterleave::ElimParameter() {
  ParallelContext::GetInstance()->get_redundancy_node().clear();
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
    ParallelContext::GetInstance()->get_redundancy_node().insert(parameter);
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

void PipelineInterleave::SetScheduler() {
  if (!is_vpp_) {
    const auto parallel_context = parallel::ParallelContext::GetInstance();
    parallel_context->set_pipeline_scheduler(parallel::kPipelineSeqpipe);
  }
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
    if (is_v_shape_ && IsOneOfPrimitiveCNode(node, {prim::kPrimMatMul, prim::kPrimBatchMatMul, prim::kPrimMatMulExt,
                                                    prim::kPrimBatchMatMulExt})) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &the_4th_input = cnode->input(kIndex4);
      const auto &trans_b = GetValueNode(the_4th_input);
      bool is_trans_b = GetValue<bool>(trans_b);
      cnode->AddPrimalAttr(FORWARD_TRANSPOSE_B, MakeValue(is_trans_b));
    }
    auto stage_info = node->user_data<NodeStageInfo>();
    if (!node->isa<CNode>() || stage_info == nullptr || stage_info->stage() == -1 ||
        IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      continue;
    }
    // Modify for lizard cyclomatic complexity.
    CutBorderForNode(shared_cell_, node, &order);
  }
  HandleSharedParam(&order);
  RemoveMonadNode();
  SetScheduler();
}

AnfNodePtr PipelinePostProcess::GetZeroOutputs(const FuncGraphPtr &graph) {
  auto real_kernel = GetRealKernelNode(graph->output(), -1);
  AnfNodePtr node = real_kernel.first;
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> out_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
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
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
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
    } else if (IsValueNode<tensor::Tensor>(pre_node)) {
      auto base_shape = pre_node->Shape();
      MS_EXCEPTION_IF_NULL(base_shape);
      auto shape_ptr = dyn_cast<abstract::Shape>(base_shape);
      MS_EXCEPTION_IF_NULL(shape_ptr);
      slice_shape = shape_ptr->shape();
    } else {
      MS_EXCEPTION_IF_NULL(pre_node->cast<CNodePtr>());
      auto op_info = pre_node->cast<CNodePtr>()->user_data<OperatorInfo>();
      MS_EXCEPTION_IF_NULL(op_info);
      std::vector<TensorInfo> tensor_info;
      if (!op_info->outputs_tensor_info_new().empty()) {
        MS_LOG(DEBUG) << "node: " << node->DebugString() << ", has tensor info new";
        auto tensor_info_new_vec = op_info->outputs_tensor_info_new();
        for (const auto &tensor_info_new : tensor_info_new_vec) {
          if (tensor_info_new->is_list()) {
            MS_LOG(EXCEPTION) << "node: " << node->DebugString() << ", its tensorinfo is list";
          }
          tensor_info.push_back(tensor_info_new->GetValue());
        }
      } else {
        tensor_info = op_info->outputs_tensor_info();
      }
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

static int64_t CalSrTag(int64_t order, int64_t micro, int64_t interleave_index, int64_t seq_chunk) {
  return order * MAX_MICRO_BATCH_NUM * MAX_INTERLEAVE_NUM + seq_chunk * MAX_SEQ_CHUNK_NUM +
         interleave_index * MAX_INTERLEAVE_NUM + micro;
}

AnfNodePtr PipelinePostProcess::GenNewNodeFromOld(const AnfNodePtr &node, const AnfNodePtr &input, int64_t micro,
                                                  int64_t index) {
  const auto &old = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto cloned_prim = prim->Clone();
  auto attrs = prim->attrs();
  auto order = GetValue<int64_t>(old->GetPrimalAttr(ORDER));
  int64_t seq_chunk = 0;
  if (old->HasPrimalAttr(SEQ_CHUNK)) {
    seq_chunk = GetValue<int64_t>(old->GetPrimalAttr(SEQ_CHUNK));
  }
  auto sr_tag = CalSrTag(order, micro, index, seq_chunk);
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
  if (old->HasPrimalAttr(FREEZE)) {
    new_node->AddPrimalAttr(FREEZE, old->GetPrimalAttr(FREEZE));
  }
  return new_node;
}

AnfNodePtr PipelinePostProcess::GenNewParamRecv(const AnfNodePtr &new_recv, const ParameterPtr &param) {
  AnfNodePtr new_node = new_recv;
  auto param_users = GetOutputNodesWithFilter(param, [&](auto anode) {
    return IsPrimitiveCNode(anode, prim::kPrimLoad) || IsPrimitiveCNode(anode, prim::kPrimCast);
  });
  for (const auto &param_user : param_users) {
    if (!IsPrimitiveCNode(param_user.first, prim::kPrimMicroStepAllGather)) {
      continue;
    }
    (void)manager_->SetEdge(param_user.first, kIndex1, new_recv);
    new_node = param_user.first;
    if (param->has_user_data(kSharedParamMirrorNode)) {
      new_node = param->user_data<AnfNode>(kSharedParamMirrorNode);
    }
  }
  return new_node;
}

std::vector<AnfNodePtr> PipelinePostProcess::GenerateMainGraphSend(const std::vector<AnfNodePtr> &nodes,
                                                                   const AnfNodePtr &node, const ValuePtr &micro,
                                                                   const ValuePtr &index) {
  std::vector<AnfNodePtr> sends;
  auto index_value = GetValue<int64_t>(index);
  int64_t send_input_idx = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto send = nodes[i];
    auto csend = send->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(csend);
    if (csend->HasPrimalAttr(PIPELINE_PARAM)) {
      if (csend->HasPrimalAttr("send_once")) {
        continue;
      }
      auto param = csend->user_data<AnfNode>(INPUT_PARAM);
      csend->AddPrimalAttr("send_once", MakeValue(true));
      auto new_send = GenNewNodeFromOld(send, param, 0, 0);
      sends.emplace_back(new_send);
      continue;
    }
    auto micro_value = GetValue<int64_t>(micro);
    auto send_input = CreateTupleGetItemNode(main_graph_, node, send_input_idx);
    MS_EXCEPTION_IF_NULL(node->cast<CNodePtr>());
    if (node->cast<CNodePtr>()->HasPrimalAttr(SEQ_CHUNK)) {
      MS_EXCEPTION_IF_NULL(send->cast<CNodePtr>());
      send->cast<CNodePtr>()->AddPrimalAttr(SEQ_CHUNK, node->cast<CNodePtr>()->GetPrimalAttr(SEQ_CHUNK));
    }
    auto new_send = GenNewNodeFromOld(send, send_input, micro_value, index_value)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(new_send);
    new_send->AddPrimalAttr(PIPELINE_END, micro);
    new_send->AddPrimalAttr(MICRO, micro);
    MS_EXCEPTION_IF_NULL(node->cast<CNodePtr>());
    if (node->cast<CNodePtr>()->HasPrimalAttr(SEQ_CHUNK)) {
      new_send->AddPrimalAttr(SEQ_CHUNK, node->cast<CNodePtr>()->GetPrimalAttr(SEQ_CHUNK));
    }
    sends.emplace_back(new_send);
    send_input_idx += 1;
  }
  return sends;
}

AnfNodePtr PipelinePostProcess::GenerateMainGraphRecv(const AnfNodePtr &fg_node, const AnfNodePtr &recv) {
  auto cuser = fg_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cuser);
  auto crecv = recv->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(crecv);
  AnfNodePtr new_recv;
  if (crecv->HasPrimalAttr(PIPELINE_PARAM)) {
    auto param = crecv->user_data<AnfNode>(INPUT_PARAM);
    MS_EXCEPTION_IF_NULL(param);
    new_recv = GenNewNodeFromOld(recv, param, 0, 0);
    auto param_node = param->cast<ParameterPtr>();
    if (param_node != nullptr) {
      const auto &recv_param_info = param_node->param_info();
      if (recv_param_info != nullptr) {
        recv_param_info->set_is_pipeline_shared_param(true);
      }
      if (NeededHandleShardParam()) {
        new_recv = GenNewParamRecv(new_recv, param_node);
      }
    }
  } else {
    auto index = cuser->GetPrimalAttr(INDEX);
    MS_EXCEPTION_IF_NULL(index);
    if (cuser->HasPrimalAttr(SEQ_CHUNK)) {
      crecv->AddPrimalAttr(SEQ_CHUNK, cuser->GetPrimalAttr(SEQ_CHUNK));
    }
    auto index_value = GetValue<int64_t>(index);
    new_recv = GenNewNodeFromOld(recv, crecv->input(1), GetValue<int64_t>(cuser->GetPrimalAttr(MICRO)), index_value);
    MS_EXCEPTION_IF_NULL(new_recv->cast<CNodePtr>());
    new_recv->cast<CNodePtr>()->AddPrimalAttr(PIPELINE_BEGIN, cuser->GetPrimalAttr(MICRO));
  }
  MS_EXCEPTION_IF_NULL(new_recv->cast<CNodePtr>());
  new_recv->cast<CNodePtr>()->AddPrimalAttr(MICRO, cuser->GetPrimalAttr(MICRO));
  if (cuser->HasPrimalAttr(SEQ_CHUNK)) {
    new_recv->cast<CNodePtr>()->AddPrimalAttr(SEQ_CHUNK, cuser->GetPrimalAttr(SEQ_CHUNK));
  }
  manager_->AddEdge(cuser, new_recv);
  return new_recv;
}

void PipelinePostProcess::Init(const std::vector<AnfNodePtr> &nodes) {
  shared_cell_ = nullptr;
  shared_cell_users_.clear();
  auto scheduler = parallel::ParallelContext::GetInstance()->pipeline_scheduler();
  if (scheduler == ZBV) {
    is_v_shape_ = true;
  }
  for (auto &node : nodes) {
    if ((IsPrimitiveCNode(node, prim::kPrimSend) || IsPrimitiveCNode(node, prim::kPrimReceive)) &&
        shared_cell_ == nullptr) {
      MS_EXCEPTION_IF_NULL(node->cast<CNodePtr>());
      shared_cell_ = node->cast<CNodePtr>()->func_graph();
    }
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      main_graph_ = graph;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    chunk_num_ = (chunk + 1) > chunk_num_ ? (chunk + 1) : chunk_num_;
  }
  auto main_graph_nodes = TopoSort(main_graph_->get_return(), SuccDeeperSimple);
  for (const auto &node : main_graph_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<FuncGraph>(cnode->input(0))) {
      continue;
    }
    auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    if (fg != shared_cell_) {
      continue;
    }
    shared_cell_users_.emplace_back(cnode);
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
    MS_EXCEPTION_IF_NULL(cnode);
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
      MS_EXCEPTION_IF_NULL(prim);
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
    if (cusr->HasPrimalAttr(SEQ_CHUNK)) {
      auto seq_chunk = GetValue<int64_t>(cusr->GetPrimalAttr(SEQ_CHUNK));
      micro_value = micro_value * MAX_MICRO_BATCH_NUM + seq_chunk;
    }
    if (!std::count(micro_visited.begin(), micro_visited.end(), micro_value)) {
      (void)micro_visited.emplace_back(micro_value);
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
  if (is_v_shape_) {
    RemoveMonadNode(fg, chunk);
  }
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
    MS_EXCEPTION_IF_NULL(cusr);
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
      (void)manager_->Replace(usr, new_usr);
    }
    main_graph_sends.insert(main_graph_sends.end(), temp_sends.begin(), temp_sends.end());
    for (auto &recv : recvs) {
      auto crecv = recv->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(crecv);
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

AnfNodePtr GetUserNode(const AnfNodePtr &node, const NodeUsersMap &node_users_map) {
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
