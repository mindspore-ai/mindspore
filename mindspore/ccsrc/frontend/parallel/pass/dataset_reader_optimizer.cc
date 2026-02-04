/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/dataset_reader_optimizer.h"
#include <algorithm>
#include <memory>
#include <list>
#include <stack>
#include <queue>
#include <string>
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/sequence_ops.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/utils/utils.h"
#include "ir/graph_utils.h"
#include "utils/tensor_construct_utils.h"
#include "frontend/parallel/pass/overlap_opt_shard_in_pipeline.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"
#include "include/utils/anfalgo.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace parallel {
constexpr auto kRankList = "rank_list";
constexpr auto kDatasetBroadcast = "dataset_broadcast";
void ControlOrder(const CNodePtr &prior, const CNodePtr &last, const FuncGraphPtr &graph,
                  const FuncGraphManagerPtr &manager, const std::string &tags) {
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), last->input(1), prior};
  auto depend = graph->NewCNode(depend_input);
  depend->set_abstract(last->input(1)->abstract());
  depend->AddPrimalAttr(tags, MakeValue(true));
  manager->SetEdge(last, 1, depend);
}

bool CreateBroadcastInput(const Shapes &shapes, const std::vector<TypePtr> &types, const FuncGraphPtr &graph,
                          AnfNodePtr *make_tuple) {
  if (shapes.size() != types.size()) {
    return false;
  }
  std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple->Clone())};
  AbstractBasePtrList make_tuple_abstract;
  for (size_t i = 0; i < shapes.size(); ++i) {
    tensor::TensorPtr zero_tensor = nullptr;
    auto cur_input_shape = shapes.at(i);
    auto cur_input_type = types.at(i);
    zero_tensor = TensorConstructUtils::CreateZerosTensor(cur_input_type, cur_input_shape);
    if (zero_tensor == nullptr) {
      return false;
    }
    auto abs = zero_tensor->ToAbstract();
    auto zero_node = NewValueNode(zero_tensor);
    zero_node->set_abstract(abs->Clone());
    make_tuple_input.emplace_back(zero_node);
    make_tuple_abstract.emplace_back(abs);
  }
  if (make_tuple_input.size() == 1) {
    return false;
  }
  *make_tuple = graph->NewCNode(make_tuple_input);
  (*make_tuple)->set_abstract(std::make_shared<abstract::AbstractTuple>(make_tuple_abstract));
  return true;
}

bool DatasetReaderOptimizer::Init() {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return false;
  }
  opt_level_ = ms_context->get_param<int>(MS_CTX_DATASET_BROADCAST_OPT_LEVEL);
  if (opt_level_ != WITHIN_STAGE && opt_level_ != OPT_ALL && opt_level_ != BETWEEN_STAGE) {
    return false;
  }
  auto is_kbk = ms_context->IsKByKExecutorMode();
  if (!is_kbk) {
    MS_LOG(WARNING) << "Now, Dataset broadcast optimize pass only support O0 and O1 jit level.";
    return false;
  }
  if (ParallelInit() != SUCCESS) {
    return false;
  }
  if (root_ == nullptr) {
    return false;
  }
  auto all_nodes = TopoSort(root_->get_return(), SuccDeeperSimple);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
      continue;
    }
    virtual_dataset_ = node;
    break;
  }
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimGetNext)) {
      continue;
    }
    get_next_ = node;
    break;
  }
  return true;
}

RankList DatasetReaderOptimizer::InferReapteDataRankThroughDataStrategy(const Strategies &data_stra) {
  auto max_shard = 1;
  RankList rank_list = {};
  if (virtual_dataset_ == nullptr) {
    return rank_list;
  }
  for (const auto &each_stra : data_stra) {
    auto cur_max_shard = *max_element(each_stra.begin(), each_stra.end());
    max_shard = (max_shard > cur_max_shard) ? max_shard : cur_max_shard;
  }
  auto dev_num = g_device_manager->stage_device_num();
  auto repeat_num = dev_num / max_shard;
  if (repeat_num == 1) {
    return rank_list;
  }
  auto cnode = virtual_dataset_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  uint64_t repeat_dim = 0;
  DeviceMatrix device_matrix;
  if (prim->HasAttr(REPEAT_DIM_DIRECT) && GetValue<std::string>(prim->GetAttr(REPEAT_DIM_DIRECT)) == RIGHT) {
    device_matrix = DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(),
                                 {max_shard, repeat_num});
    repeat_dim = 1;
  } else {
    device_matrix = DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(),
                                 {repeat_num, max_shard});
  }
  if (device_matrix.GetDevicesAlongDim(repeat_dim, &rank_list)) {
    MS_LOG(WARNING) << "Failed to get dataset repeat rank list within pipeline stage.";
  }
  return rank_list;
}

std::vector<RankList> DatasetReaderOptimizer::InferRepeatDataRankThroughLayout() {
  // ((2, 2, 2),)
  auto all_dev_mat = ParallelContext::GetInstance()->dataset_strategy_devmat();
  // (((2), (1), (0)),)
  auto all_tensor_map = ParallelContext::GetInstance()->dataset_strategy_tensormap();
  std::vector<RankList> all_rank_list;
  std::vector<int64_t> last_repeated_dim;
  for (size_t idx = 0; idx < all_dev_mat.size(); ++idx) {
    RankList rank_list = {};
    if (virtual_dataset_ == nullptr) {
      all_rank_list.push_back(rank_list);
      continue;
    }
    auto tensor_map = all_tensor_map.at(idx);
    auto dev_mat = all_dev_mat.at(idx);
    std::vector<int64_t> used_dev_idx = {};
    std::vector<int64_t> repeated_dim = {};
    for (size_t i = 0; i < tensor_map.size(); ++i) {
      for (size_t j = 0; j < tensor_map.at(i).size(); ++j) {
        auto tensor_map_value = tensor_map.at(i).at(j);
        if (tensor_map_value != -1) {
          auto real_idx = dev_mat.size() - LongToSize(tensor_map_value) - 1;
          used_dev_idx.push_back(SizeToLong(real_idx));
        }
      }
    }
    for (size_t i = 0; i < dev_mat.size(); ++i) {
      if (std::find(used_dev_idx.begin(), used_dev_idx.end(), i) == used_dev_idx.end()) {
        repeated_dim.push_back(SizeToLong(i));
      }
    }
    if (!repeated_dim.empty()) {
      DeviceMatrix device_matrix =
        DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(), dev_mat);
      device_matrix.GetDevicesAlongMultiDim(repeated_dim, &rank_list);
    }
    if (!all_rank_list.empty()) {
      if (!std::equal(repeated_dim.begin(), repeated_dim.end(), last_repeated_dim.begin())) {
        MS_LOG(EXCEPTION) << "The repeated dim for each layout must be equal, but got current repeated_dim "
                          << repeated_dim << ", last repeated_dim " << last_repeated_dim;
      }
    }
    last_repeated_dim = repeated_dim;
    all_rank_list.push_back(rank_list);
  }
  return all_rank_list;
}

std::vector<RankList> DatasetReaderOptimizer::InferRepeatRankListWithinStage() {
  std::vector<RankList> rank_list = {{}};
  if (opt_level_ != WITHIN_STAGE && opt_level_ != OPT_ALL) {
    return rank_list;
  }

  if (!ParallelContext::GetInstance()->dataset_strategy_tensormap().empty() &&
      !ParallelContext::GetInstance()->dataset_strategy_devmat().empty()) {
    return InferRepeatDataRankThroughLayout();
  }

  auto data_stra = ParallelContext::GetInstance()->dataset_strategy();
  if (!data_stra.empty()) {
    return {InferReapteDataRankThroughDataStrategy(data_stra)};
  }
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  if (full_batch) {
    return {g_device_manager->GetDeviceListInThisStage()};
  }
  return rank_list;
}

AnfNodePtr DatasetReaderOptimizer::FindDatasetParameter(const AnfNodePtr &node, const NodeUsersMap &node_users_map) {
  std::stack<AnfNodePtr> st;
  HashSet<AnfNodePtr> visited;
  st.push(node);
  visited.insert(node);
  while (!st.empty()) {
    auto cur_node = st.top()->cast<CNodePtr>();
    st.pop();
    auto cur_node_users = node_users_map.at(cur_node);
    for (const auto &node_pair : cur_node_users) {
      auto user_node = node_pair.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_node);
      if (IsValueNode<FuncGraph>(user_node->input(0))) {
        auto fg = GetValueNode<FuncGraphPtr>(user_node->input(0));
        MS_EXCEPTION_IF_NULL(fg);
        if (fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
          auto fg_params = fg->parameters();
          return fg_params.at(node_pair.second - 1);
        }
      }
      if (visited.find(user_node) == visited.end() && !IsPrimitiveCNode(user_node, prim::kPrimReturn)) {
        st.push(user_node);
        visited.insert(user_node);
      }
    }
  }
  return nullptr;
}

void DatasetReaderOptimizer::FindAllStageIdUsedDataParameter(const AnfNodePtr &node, const NodeUsersMap &node_users_map,
                                                             std::set<int64_t> *const data_used_stage) {
  if (opt_level_ != BETWEEN_STAGE && opt_level_ != OPT_ALL) {
    return;
  }
  std::queue<AnfNodePtr> queue;
  HashSet<AnfNodePtr> visited;
  queue.push(node);
  visited.insert(node);
  while (!queue.empty()) {
    auto cur_node = queue.front();
    queue.pop();
    auto cur_node_users = node_users_map.at(cur_node);
    for (const auto &node_pair : cur_node_users) {
      auto user_node = node_pair.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_node);
      if (IsValueNode<FuncGraph>(user_node->input(0))) {
        auto fg = GetValueNode<FuncGraphPtr>(user_node->input(0));
        MS_EXCEPTION_IF_NULL(fg);
        auto stage = fg->stage();
        if (stage != -1) {
          data_used_stage->insert(stage);
        }
        continue;
      }
      if (visited.find(user_node) == visited.end() && !IsPrimitiveCNode(user_node, prim::kPrimReturn)) {
        queue.push(user_node);
        visited.insert(user_node);
      }
    }
  }
  return;
}

RankList DatasetReaderOptimizer::InferRepeatRankList(const RankList &within_stage, const RankList &between_stage) {
  RankList rank_list;
  auto local_stage = g_device_manager->stage_id();
  auto stage_device_num = g_device_manager->stage_device_num();
  if (between_stage.empty()) {
    return within_stage;
  }
  if (within_stage.empty()) {
    std::transform(between_stage.begin(), between_stage.end(), std::back_inserter(rank_list),
                   [&](const auto &stage_id) {
                     auto global_rank = g_device_manager->global_rank();
                     auto tar_rank = global_rank + SizeToLong(stage_device_num) * (stage_id - local_stage);
                     return tar_rank;
                   });
    return rank_list;
  }
  for (const auto &stage_id : between_stage) {
    std::transform(within_stage.begin(), within_stage.end(), std::back_inserter(rank_list),
                   [&](const auto &repeat_rank) {
                     auto tar_rank = repeat_rank + SizeToLong(stage_device_num) * (stage_id - local_stage);
                     return tar_rank;
                   });
  }
  return rank_list;
}

void DatasetReaderOptimizer::InsertBroadcast(const RankList &rank_list) {
  auto global_rank = g_device_manager->global_rank();
  auto iter = std::find(rank_list.begin(), rank_list.end(), global_rank);
  if (iter == rank_list.end()) {
    return;
  }
  std::vector<AnfNodePtr> broadcast_input = {NewValueNode(prim::kPrimBroadcast->Clone())};
  (void)broadcast_input.emplace_back(get_next_);
  auto broadcast = root_->NewCNode(broadcast_input);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), get_next_, broadcast};
  auto depend = root_->NewCNode(depend_input);
  (void)manager_->Replace(get_next_, depend);
  Group data_repeat_group;
  if (g_device_manager->CreateGroup(rank_list, &data_repeat_group) != SUCCESS) {
    MS_LOG(WARNING) << "Create dataset repeat group failed, rank list is: " << rank_list;
    return;
  }
  if (broadcast == nullptr) {
    return;
  }
  auto prim = GetCNodePrimitive(broadcast);
  MS_EXCEPTION_IF_NULL(prim);
  prim->set_attr(ROOT_RANK, MakeValue(BROADCAST_ROOT_RANK));
  prim->set_attr(GROUP, MakeValue(data_repeat_group.name()));
  prim->set_attr(DATASET_BROADCAST, MakeValue(True));
  prim->set_attr(GRAPH_FLAG_SIDE_EFFECT_MEM, MakeValue<bool>(True));
  // Broadcast's first input will be inplace updated.
  // Because primitive created from C++ prim does not have extra signature info, we need to set rw_write manually.
  prim->set_rw_write_input_indexes({kIndex0});

  broadcast->AddAttr(kDatasetBroadcast, MakeValue<bool>(true));
  depend->AddAttr(kDatasetBroadcast, MakeValue<bool>(true));

  auto rank_list_ptr = std::make_shared<RankList>(rank_list);
  broadcast->set_user_data<RankList>(kRankList, rank_list_ptr);
  CNodePtr get_next_c = get_next_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(get_next_c);
  get_next_c->AddAttr(kDatasetBroadcast, MakeValue<bool>(true));
}

void DatasetReaderOptimizer::BroadcastDataset() {
  if (get_next_ == nullptr) {
    MS_LOG(WARNING) << "For now on, only dataset sink mode support dataset reader optimizer.";
    return;
  }
  if (virtual_dataset_ == nullptr) {
    return;
  }
  auto reapte_rank_within_stage = InferRepeatRankListWithinStage().at(0);
  const auto &node_users_map = manager_->node_users();
  auto dataset_users = node_users_map.at(virtual_dataset_);
  std::set<int64_t> data_used_stage;
  std::vector<int64_t> tuple_get_item_idx = {};
  for (const auto &node_pair : dataset_users) {
    auto cnode = node_pair.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto cur_input_parameter = FindDatasetParameter(cnode, node_users_map);
    if (cur_input_parameter != nullptr) {
      FindAllStageIdUsedDataParameter(cur_input_parameter, node_users_map, &data_used_stage);
      tuple_get_item_idx.push_back(GetTupleGetItemIndex(cnode));
    }
  }
  RankList reapte_rank_between_stage;
  reapte_rank_between_stage.assign(data_used_stage.begin(), data_used_stage.end());
  auto rank_list = InferRepeatRankList(reapte_rank_within_stage, reapte_rank_between_stage);
  if (rank_list.size() <= 1) {
    return;
  }
  InsertBroadcast(rank_list);
}

void InsertDepend(const std::vector<CNodePtr> &nodes, const FuncGraphPtr &graph, const FuncGraphManagerPtr &manager) {
  size_t size = nodes.size();
  if (size <= 1) {
    return;
  }
  for (size_t i = 0; i < size - 1; i++) {
    const auto &prior = nodes[i];
    const auto &last = nodes[i + 1];
    ControlOrder(prior, last, graph, manager, kDatasetBroadcast);
  }
}

void FreezeParallelOptimizerCommOrder(const FuncGraphPtr &graph) {
  bool find_dataset_broadcast = false;
  auto all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->HasAttr(kDatasetBroadcast)) {
      continue;
    }
    auto attr = cnode->GetAttr(kDatasetBroadcast);
    MS_EXCEPTION_IF_NULL(attr);
    if (GetValue<bool>(attr)) {
      find_dataset_broadcast = true;
      break;
    }
  }
  if (!find_dataset_broadcast) {
    return;
  }
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<CNodePtr> allgather_vec;
  std::vector<CNodePtr> reducescatter_vec;
  std::list<CNodePtr> graph_orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(graph_orders.begin(), graph_orders.end());
  for (const auto &node : origin_nodes_topological) {
    if (IsPrimitiveCNode(node, prim::kPrimAllGather) && common::AnfAlgo::IsFromParallelOptimizer(node)) {
      allgather_vec.push_back(node);
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimReduceScatter) && common::AnfAlgo::IsFromParallelOptimizer(node)) {
      reducescatter_vec.push_back(node);
    }
  }
  InsertDepend(allgather_vec, graph, manager);
  InsertDepend(reducescatter_vec, graph, manager);
}

void ReplaceGetnextWithBroadcast(const FuncGraphPtr &graph) {
  auto all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  CNodePtr broadcast = nullptr;
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->HasAttr(kDatasetBroadcast)) {
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimBroadcast)) {
      broadcast = cnode;
      break;
    }
  }
  if (broadcast == nullptr) {
    return;
  }
  if (!broadcast->has_user_data(kRankList)) {
    return;
  }
  auto rank_list = broadcast->user_data<RankList>(kRankList);
  MS_EXCEPTION_IF_NULL(rank_list);
  auto global_rank = g_device_manager->global_rank();
  auto iter = std::find(rank_list->begin(), rank_list->end(), global_rank);
  if (iter == rank_list->begin()) {
    return;
  }
  CNodePtr get_next = broadcast->input(1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(get_next);
  if (!IsPrimitiveCNode(get_next, prim::kPrimGetNext)) {
    MS_LOG(EXCEPTION) << "For ReplaceGetnextWithBroadcast: found Broadcast, but not GetNext. ";
  }
  const auto &manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users_map = manager->node_users();
  auto users = node_users_map.at(broadcast);
  CNodePtr depend = users.front().first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend);

  auto prim = GetCNodePrimitive(get_next);
  MS_EXCEPTION_IF_NULL(prim);
  auto shape_attr = prim->GetAttr(SHAPES);
  auto type_attr = prim->GetAttr(TYPES);
  if (shape_attr == nullptr || type_attr == nullptr) {
    return;
  }

  std::vector<ValuePtr> shape;
  if (shape_attr->isa<ValueTuple>()) {
    const auto &value_tuple = shape_attr->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    shape = value_tuple->value();
  } else {
    const auto &value_list = shape_attr->cast<ValueListPtr>();
    MS_EXCEPTION_IF_NULL(value_list);
    shape = value_list->value();
  }
  Shapes shapes;
  for (const auto &element : shape) {
    std::vector<ValuePtr> element_list;
    if (element->isa<ValueTuple>()) {
      const auto &value_tuple = element->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      element_list = value_tuple->value();
    } else {
      const auto &value_list = element->cast<ValueListPtr>();
      MS_EXCEPTION_IF_NULL(value_list);
      element_list = value_list->value();
    }
    Shape shape_vec;
    (void)std::transform(element_list.begin(), element_list.end(), std::back_inserter(shape_vec),
                         [](const ValuePtr &v) -> int64_t { return GetValue<int64_t>(v); });
    shapes.emplace_back(shape_vec);
  }
  auto types = GetValue<std::vector<TypePtr>>(type_attr);
  AnfNodePtr real_input = nullptr;
  if (!CreateBroadcastInput(shapes, types, graph, &real_input)) {
    return;
  }
  manager->SetEdge(broadcast, 1, real_input);
  broadcast->set_abstract(get_next->abstract()->Clone());
  manager->Replace(get_next, broadcast);
  manager->Replace(depend, depend->input(1));
}

void ControlOptShardCommAndDataBroadcastOrder(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return;
  }
  auto opt_level = ms_context->get_param<int>(MS_CTX_DATASET_BROADCAST_OPT_LEVEL);
  if (opt_level == 0) {
    return;
  }
  if (graph == nullptr) {
    return;
  }
  if (!IsTraining(graph->manager())) {
    return;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    return;
  }
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    return;
  }
  auto all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  CNodePtr broadcast_op;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimBroadcast)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    if (!prim->HasAttr(DATASET_BROADCAST)) {
      continue;
    }
    broadcast_op = cnode;
    break;
  }
  if (broadcast_op == nullptr) {
    return;
  }
  std::vector<CNodePtr> opt_shard_comm_list;
  for (const auto &node : all_nodes) {
    if (!is_allgather_comm_ops(node)) {
      continue;
    }
    auto all_gather_cnode = node->cast<CNodePtr>();
    (void)opt_shard_comm_list.emplace_back(all_gather_cnode);
  }
  if (opt_shard_comm_list.empty()) {
    return;
  }
  for (const auto &opt_shard_comm : opt_shard_comm_list) {
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), opt_shard_comm->input(1), broadcast_op};
    auto depend_node = graph->NewCNode(depend_inputs);
    depend_node->set_abstract(opt_shard_comm->input(1)->abstract()->Clone());
    (void)manager->Replace(opt_shard_comm->input(1), depend_node);
  }
  return;
}

static std::vector<CNodePtr> GetPPComms(const AnfNodePtrList &nodes) {
  std::vector<CNodePtr> pp_comms;
  for (const auto &node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimReceive) && !IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }
    if (is_first_receive(node)) {
      pp_comms.emplace_back(node->cast<CNodePtr>());
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      continue;
    }
    if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      continue;
    }
    pp_comms.emplace_back(cnode);
  }
  return pp_comms;
}

void ControlPipelineCommAndDataBroadcastOrder(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return;
  }
  if (ms_context->get_param<int>(MS_CTX_DATASET_BROADCAST_OPT_LEVEL) == 0) {
    return;
  }
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (g_device_manager == nullptr) {
    return;
  }
  auto stage_num = g_device_manager->stage_num();
  if (stage_num <= 1) {
    return;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    return;
  }
  auto all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  auto pp_comms = GetPPComms(all_nodes);
  if (pp_comms.empty()) {
    return;
  }
  CNodePtr broadcast_op;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimBroadcast)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    if (!prim->HasAttr(DATASET_BROADCAST)) {
      continue;
    }
    broadcast_op = cnode;
    break;
  }
  if (broadcast_op == nullptr) {
    return;
  }
  for (const auto &pp_comm : pp_comms) {
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), pp_comm->input(1), broadcast_op};
    auto depend_node = graph->NewCNode(depend_inputs);
    depend_node->set_abstract(pp_comm->input(1)->abstract()->Clone());
    (void)manager->Replace(pp_comm->input(1), depend_node);
  }
}
}  // namespace parallel
}  // namespace mindspore
